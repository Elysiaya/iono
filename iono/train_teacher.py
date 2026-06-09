# train_teacher.py — FGL（未来引导学习）分阶段训练脚本
# Phase 1: 教师模型训练（输入 = 历史 + 未来特权信息）
# Phase 2: 学生模型训练（输入 = 仅历史，通过引导损失模仿教师隐状态）

import os
import math
from scripts import send_email
import logging
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"train_teacher_{timestamp}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from iono.model_fgl import TeacherForecaster
from iono.config import Config
from iono.dataloader import build_dataloaders



# ==================== Phase 1: 教师模型训练 ====================

def train_teacher():
    """Train teacher model with privileged future TEC, output Config.pred_steps hours."""
    Config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Config.checkpoints_dir / f"teacher_fgl_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Config.logs_dir / "teacher"))

    train_loader, val_loader = build_dataloaders(
        Config.hickle_paths,
        Config.window_size,
        Config.future_size,
        Config.pred_steps,
        Config.batch_size,
    )

    # 教师模型：带 FiLM 特权注入的 Encoder-Decoder ConvLSTM
    teacher = TeacherForecaster(
        in_channels=Config.in_channels,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        num_aux=5,  # [新增日周期] Kp, Dst, F10.7, doy_sin, doy_cos
        pred_steps=Config.pred_steps,
        priv_gru_hidden=Config.priv_gru_hidden
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"🔥 启用 {torch.cuda.device_count()} 张 GPU 并行训练！")
        teacher = nn.DataParallel(teacher)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=Config.lr_decay_factor, patience=Config.lr_decay_patience, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda')  # AMP 混合精度

    best_val_loss = float('inf')
    patience = Config.early_stop_patience  # Early Stopping 耐心值
    patience_counter = 0

    # 计算计划采样衰减的周期（例如在前 80% 的 epoch 中将 tf_ratio 逐渐从 1.0 降到 0.0）
    decay_epochs = max(1, int(Config.num_epochs * 0.8))

    for epoch in range(Config.num_epochs):
        
        # 恢复经典的纯线性衰减，让 Teacher 更加平缓地向自回归过渡
        current_tf_ratio = max(0.0, 1.0 - (epoch / decay_epochs))

        # ---- 训练（AMP 混合精度） ----
        teacher.train()
        train_loss = 0.0
        valid_batches = 0  # [新增] 记录有效 batch
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.num_epochs}] Train")
        for batch_idx, (X_hist, aux_hist, X_future, aux_future, y) in enumerate(train_pbar):
            X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
            X_future, aux_future = X_future.to(device), aux_future.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # 优化点 3: 传入 y_true 和 tf_ratio 激活计划采样
                pred = teacher(
                    X_hist, 
                    aux_x=aux_hist, 
                    future_tec=X_future, 
                    future_aux=aux_future,
                    dec_aux=aux_future[:, :Config.pred_steps, :], 
                    y_true=y, 
                    tf_ratio=current_tf_ratio
                )
                loss = criterion(pred, y)

            # 优化点 4: 异常值截断保护
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                logger.warning("发现 NaN/Inf Loss，跳过该 Batch 保护模型！")
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            valid_batches += 1
            
            # 实时更新进度条显示的 Loss
            train_pbar.set_postfix({'Loss(RMSE)': f"{loss.item()**0.5 * 100:.4f}"})
            
            # 偶尔记录到日志文件（每 50 个 batch），绕过终端打断进度条
            if valid_batches % 50 == 0:
                msg = f"Epoch [{epoch+1}/{Config.num_epochs}] Batch [{valid_batches}/{len(train_loader)}] Loss: {loss.item()**0.5 * 100:.4f}"
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.emit(logging.LogRecord(logger.name, logging.INFO, '', 0, msg, (), None))
                
        if valid_batches == 0:
            logger.error("整个 Epoch 的 Loss 都是 NaN，训练终止！请检查数据异常。")
            break

        avg_train = train_loss / valid_batches

        # ---- 验证 ----
        teacher.eval()
        val_loss = 0.0
        step_mse = torch.zeros(Config.pred_steps, device=device)
        total_samples = 0
        
        # 优化点 5: 验证集也加上 AMP 混合精度，节省显存并提速
        with torch.no_grad(), torch.amp.autocast('cuda'):
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{Config.num_epochs}] Val  ", leave=False)
            for X_hist, aux_hist, X_future, aux_future, y in val_pbar:
                X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
                X_future, aux_future = X_future.to(device), aux_future.to(device)
                y = y.to(device)
                B = y.size(0)
                total_samples += B

                # 验证时不传 y_true 和 tf_ratio，强制模型使用 100% 自回归进行推断
                pred = teacher(X_hist, aux_x=aux_hist, future_tec=X_future, future_aux=aux_future, dec_aux=aux_future[:, :Config.pred_steps, :])
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                # [新增] 计算每步的高精度 RMSE 统计
                for t in range(Config.pred_steps):
                    step_mse[t] += nn.functional.mse_loss(pred[:, t], y[:, t]).item() * B
                
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        model_state = teacher.module.state_dict() if isinstance(teacher, nn.DataParallel) else teacher.state_dict()
        
        val_rmse = (avg_val ** 0.5) * 100.0
        train_rmse = (avg_train ** 0.5) * 100.0
        step_rmse = torch.sqrt(step_mse / total_samples) * 100.0
        
        # ---- 保存 ----
        checkpoint = {
            'epoch': epoch + 1,
            'config': Config.training_snapshot(),
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
        }
        name = f"teacher_epoch{epoch+1:02d}_valRMSE{val_rmse:.4f}_trainRMSE{train_rmse:.4f}.pth"
        torch.save(checkpoint, save_dir / name)
        if epoch % 5 == 0:  # 每 5 个 epoch 发送一次邮件通知
            send_email.send_email(f"Teacher model Epoch {epoch+1} completed! Val RMSE: {val_rmse:.4f}, Train RMSE: {train_rmse:.4f}. Model saved as {name}.")
        

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(checkpoint, save_dir / "best_teacher.pth")
            torch.save(checkpoint, Config.teacher_checkpoint)
            logger.info(f"  --> Saved new BEST teacher: {name}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  --> Early Stopping at epoch {epoch+1}! Val loss has not improved for {patience} consecutive epochs.")
                break

        # [修改] 对齐学生模型的精美日志打印格式
        logger.info(f"[Teacher] Epoch [{epoch+1}/{Config.num_epochs}] TF_Ratio: {current_tf_ratio:.2f}")
        logger.info(f"          Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
        step_rmse_str = ", ".join([f"{val:.2f}" for val in step_rmse.tolist()])
        logger.info(f"          Val Step 1-{Config.pred_steps}: [{step_rmse_str}]")
        logger.info(f"          Best Val RMSE: {(best_val_loss ** 0.5) * 100:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        writer.add_scalar('Teacher/train_loss', avg_train, epoch)
        writer.add_scalar('Teacher/val_loss', avg_val, epoch)
        writer.add_scalar('Teacher/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Teacher/tf_ratio', current_tf_ratio, epoch)

    writer.close()
    logger.info(f"Teacher training finished! Checkpoints saved in {save_dir}")
    content = "Teacher model training completed! Please check the training logs and saved model files."
    send_email.send_email(content)
    return str(save_dir / "best_teacher.pth")


# ==================== 主入口 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 1: Training Teacher Model (Historical + Future Privileged Information)")
    logger.info("=" * 60)
    train_teacher()
