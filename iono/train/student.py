import os
import math
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from iono.model_fgl import TeacherForecaster,StudentForecaster
from iono.config import Config
from iono.training import build_temporal_dataloaders
from scripts.send_email import send_email
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_student.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def build_dataloaders(hickle_paths, window_size, future_size, pred_steps, batch_size):
    return build_temporal_dataloaders(
        hickle_paths,
        window_size,
        future_size,
        pred_steps,
        batch_size,
        logger=logger,
    )

# ==================== Phase 2: 学生模型训练（FGL 蒸馏） ====================
def train_student():
    """Train student model: input 72h history only, mimic teacher hidden state."""
    Config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Config.checkpoints_dir / f"student_fgl_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Config.logs_dir / "student"))

    train_loader, val_loader = build_dataloaders(
        Config.hickle_paths,
        Config.window_size,
        Config.future_size,
        Config.pred_steps,
        Config.batch_size,
    )

    # ---- 加载冻结教师模型 ----
    teacher = TeacherForecaster(
        in_channels=Config.in_channels,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        num_aux=5,
        pred_steps=Config.pred_steps,
        priv_gru_hidden=Config.priv_gru_hidden,
    ).to(device)
    teacher_ckpt = torch.load(Config.teacher_checkpoint, map_location=device, weights_only=True)
    # [注意] 如果之前有维度不匹配的报错，请确保 Config 中的参数与 teacher 训练时绝对一致
    teacher.load_state_dict(teacher_ckpt['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)
    logger.info(f"Loaded teacher from: {Config.teacher_checkpoint}")

    # ---- 初始化学生模型 ----
    student = StudentForecaster(
        in_channels=Config.in_channels,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        num_aux=5,
        pred_steps=Config.pred_steps,
        priv_gru_hidden=Config.priv_gru_hidden,
    ).to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"🔥 启用 {torch.cuda.device_count()} 张 GPU 并行训练！")
        student = nn.DataParallel(student)

    criterion_pred = nn.MSELoss()
    criterion_guide = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=Config.learning_rate)
    
    # [修改] 学习率衰减策略调整：增加一点耐心，降低衰减幅度，防止学习率过早崩溃
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda')  # AMP 混合精度

    base_lam = Config.lam  # 初始引导损失权重 λ
    best_val_loss = float('inf')
    start_epoch = 0
    patience = 12  # Early Stopping 耐心值稍微放大一点
    patience_counter = 0

    resume_ckpt = Config.resume_ckpt_student  # TODO: 填入此时表现最好的 Epoch 8 模型路径，例如："checkpoints/xxx/student_epoch08_valRMSE5.3648_trainRMSE2.1744.pth"
    # ---- 断点继续训练逻辑 ----
    if resume_ckpt and os.path.exists(resume_ckpt):
        logger.info(f"准备从断点继续训练: {resume_ckpt}")
        logger.info(f"Loading resume checkpoint from: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        student_state = student.module if isinstance(student, nn.DataParallel) else student
        student_state.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_pred_loss', float('inf'))
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch}, best val pred loss: {best_val_loss:.6f}")

    for epoch in range(start_epoch, Config.num_epochs):
        # ---- 计算当前 epoch 的 teacher forcing 比例 ----
        if epoch < Config.tf_decay_epochs:
            tf_ratio = Config.tf_start_ratio - (Config.tf_start_ratio - Config.tf_end_ratio) * (epoch / Config.tf_decay_epochs)
        else:
            tf_ratio = Config.tf_end_ratio

        # [修改] 设置 TF_ratio 的保底下限，避免长序列预测在完全无引导下彻底崩溃
        tf_ratio = max(0.3, tf_ratio)

        # [修改] 固定蒸馏权重 λ：不再随 tf_ratio 同步衰减。
        # 当模型越来越依赖自身输出时，更加需要 Teacher 强势纠偏
        current_lam = base_lam

        # [新增] 时间衰减权重（Time-weighted loss）：前几个 step 权重高（促使短期误差被强力优化），向后递减
        time_weights = torch.linspace(1.5, 0.5, steps=Config.pred_steps, device=device)

        # ---- 训练（AMP 混合精度） ----
        student.train()
        train_pred_loss = 0.0
        train_guide_loss = 0.0
        train_total_loss = 0.0
        valid_batches = 0 # [新增] 用于记录有效 batch

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.num_epochs}] Train")
        for batch_idx, (X_hist, aux_hist, X_future, aux_future, y) in enumerate(train_pbar):
            
            X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
            X_future, aux_future = X_future.to(device), aux_future.to(device)
            y = y.to(device)

            # Teacher forward (no grad)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                pred_T, H_T = teacher(
                    X_hist,
                    aux_x=aux_hist,
                    future_tec=X_future,
                    future_aux=aux_future,
                    dec_aux=aux_future[:, :Config.pred_steps, :],
                    return_hidden=True,
                )

            # Student forward
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_S, H_S = student(
                    X_hist, 
                    aux_x=aux_hist,
                    future_aux = aux_future,
                    dec_aux=aux_future[:, :Config.pred_steps, :],
                    return_hidden=True,
                    y_true=y, 
                    tf_ratio=tf_ratio
                )
                
                # [修改] 使用带有时间衰减权重的预测损失，强制网络压低前几个小时（短期）的预测误差
                # criterion_pred(pred_S, y) 计算出的 shape 一般会先在 batch 上 mean，所以这里将时间维解开做标量乘法更清晰
                # pred_S 和 y 的 shape 通常是 (B, T, C, H, W) 或 (B, T, H, W)
                # 为确保维度对齐，我们在除了时间维(dim=1)以外的所有维度上求平均
                # 这样 mse_per_step 的 shape 会变成 (T,)
                mse_per_step = torch.mean((pred_S - y) ** 2, dim=(0, 2, 3, 4) if pred_S.dim() == 5 else (0, 2, 3))
                L_pred = torch.mean(mse_per_step * time_weights)

                L_guide = 0.0
                for h_s, h_t in zip(H_S, H_T):
                    # [修改] 在引导损失中同时加入 MSE 和 Cosine 相似度，促使量级和方向特征全面对齐老师
                    mse_g = criterion_guide(h_s, h_t)
                    # 展平隐状态，算各个 batch 的 cosine similarity
                    cos_g = 1.0 - nn.functional.cosine_similarity(h_s.view(h_s.size(0), -1), h_t.view(h_t.size(0), -1)).mean()
                    L_guide += mse_g + 0.5 * cos_g
                
                # [新增] Soft-Target KD：拟合 Teacher 平滑且去噪的预测输出
                L_soft = nn.functional.mse_loss(pred_S, pred_T.detach())
                alpha_soft = 0.5
                
                L_total = L_pred + current_lam * L_guide + alpha_soft * L_soft

            # [新增] 异常值阻断：如果 Loss 爆炸（NaN 或 Inf），直接跳过这个 batch，保护模型
            if math.isnan(L_total.item()) or math.isinf(L_total.item()):
                logger.warning("发现 NaN/Inf Loss，跳过该 Batch！")
                continue

            scaler.scale(L_total).backward()
            scaler.unscale_(optimizer)
            
            # [修改] 收紧梯度裁剪阈值，从 1.0 改为 0.5，进一步抑制突刺
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5) 
            
            scaler.step(optimizer)
            scaler.update()

            train_pred_loss += L_pred.item()
            train_guide_loss += L_guide.item()
            train_total_loss += L_total.item()
            valid_batches += 1
            
            # 实时更新进度条
            train_pbar.set_postfix({'TotLoss': f"{L_total.item():.4f}", 'PredRMSE': f"{L_pred.item()**0.5 * 100:.4f}"})
            
            # 偶尔记录到日志文件（每 50 个 batch）
            if valid_batches % 50 == 0:
                msg = (f"Epoch [{epoch+1}/{Config.num_epochs}] Batch [{valid_batches}/{len(train_loader)}] "
                       f"Total Loss: {L_total.item():.4f} | Pred RMSE: {L_pred.item()**0.5 * 100:.4f}")
                # 绕过终端输出，只寻找 FileHandler 写入日志文件
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.emit(logging.LogRecord(logger.name, logging.INFO, '', 0, msg, (), None))

        if valid_batches == 0:
            logger.error("整个 Epoch 的 Loss 都是 NaN，训练终止！请检查学习率是否过大或数据是否包含异常值。")
            break

        avg_pred = train_pred_loss / valid_batches
        avg_guide = train_guide_loss / valid_batches
        avg_total = train_total_loss / valid_batches

        # ---- 验证（仅用预测损失评估学生真实能力） ----
        student.eval()
        val_pred_loss = 0.0
        val_guide_loss = 0.0
        step_mse = torch.zeros(Config.pred_steps, device=device)
        total_samples = 0
        with torch.no_grad(), torch.amp.autocast('cuda'):
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{Config.num_epochs}] Val  ", leave=False)
            for X_hist, aux_hist, X_future, aux_future, y in val_pbar:
                X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
                X_future, aux_future = X_future.to(device), aux_future.to(device)
                y = y.to(device)
                B = y.size(0)
                total_samples += B
                
                _, H_T = teacher(
                    X_hist,
                    aux_x=aux_hist,
                    future_tec=X_future,
                    future_aux=aux_future,
                    dec_aux=aux_future[:, :Config.pred_steps, :],
                    return_hidden=True,
                )

                pred_S, H_S = student(
                    X_hist, 
                    aux_x=aux_hist, 
                    future_aux = aux_future,
                    dec_aux=aux_future[:, :Config.pred_steps, :], 
                    return_hidden=True,
                    tf_ratio=0.0
                )
                val_pred_loss += criterion_pred(pred_S, y).item()
                
                # [新增] 计算每步的高精度 RMSE 统计
                for t in range(Config.pred_steps):
                    step_mse[t] += nn.functional.mse_loss(pred_S[:, t], y[:, t]).item() * B
                
                batch_guide_loss = 0.0
                for h_s, h_t in zip(H_S, H_T):
                    batch_guide_loss += criterion_guide(h_s, h_t).item()
                val_guide_loss += batch_guide_loss

        nv = len(val_loader)
        avg_val_pred = val_pred_loss / nv
        avg_val_guide = val_guide_loss / nv
        avg_val_total = avg_val_pred + current_lam * avg_val_guide

        scheduler.step(avg_val_pred)  # 用预测损失调度学习率

        val_rmse = (avg_val_pred ** 0.5) * 100.0
        train_rmse = (avg_pred ** 0.5) * 100.0
        
        # 计算每个预测步长的单独 RMSE
        step_rmse = torch.sqrt(step_mse / total_samples) * 100.0
        
        student_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()

        # ---- 保存 ----
        checkpoint = {
            'epoch': epoch + 1,
            'config': Config.training_snapshot(),
            'model_state_dict': student_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_pred_loss': avg_pred,
            'train_guide_loss': avg_guide,
            'train_total_loss': avg_total,
            'val_pred_loss': avg_val_pred,
            'val_guide_loss': avg_val_guide,
            'lambda': current_lam, # [修改] 记录当前的动态 lambda
        }
        name = (f"student_epoch{epoch+1:02d}"
                f"_valRMSE{val_rmse:.4f}"
                f"_trainRMSE{train_rmse:.4f}.pth")
        
        # 为了节省硬盘空间，日常 epoch 可选择不保存或者定期覆盖，这里保持你原有的逻辑
        torch.save(checkpoint, save_dir / name)

        if avg_val_pred < best_val_loss:
            best_val_loss = avg_val_pred
            torch.save(checkpoint, save_dir / "best_student.pth")
            torch.save(checkpoint, Config.student_checkpoint)
            logger.info(f"  --> Saved new BEST student: {name}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  --> Early Stopping at epoch {epoch+1}! Val loss 已连续 {patience} 个 epoch 未改善。")
                break

        # [修改] 打印日志增加了当前 lam 值的显示，方便你观察变化
        logger.info(f"[Student] Epoch [{epoch+1}/{Config.num_epochs}]  TF_ratio={tf_ratio:.3f} | λ={current_lam:.3f}")
        logger.info(f"          Train RMSE: {train_rmse:.4f} (guide_loss: {avg_guide:.4f})")
        logger.info(f"          Val RMSE: {val_rmse:.4f} (guide_loss: {avg_val_guide:.4f})")
        
        # [新增] 打印每个预测步长的预测 RMSE
        step_rmse_str = ", ".join([f"{val:.2f}" for val in step_rmse.tolist()])
        logger.info(f"          Val Step 1-{Config.pred_steps} RMSE: [{step_rmse_str}]")
        
        logger.info(f"          Best Val RMSE: {(best_val_loss ** 0.5) * 100.0:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        writer.add_scalar('Student/train_pred_loss', avg_pred, epoch)
        writer.add_scalar('Student/train_guide_loss', avg_guide, epoch)
        writer.add_scalar('Student/train_total_loss', avg_total, epoch)
        writer.add_scalar('Student/val_pred_loss', avg_val_pred, epoch)
        writer.add_scalar('Student/val_guide_loss', avg_val_guide, epoch)
        writer.add_scalar('Student/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Student/tf_ratio', tf_ratio, epoch)
        writer.add_scalar('Student/lambda', current_lam, epoch)

    writer.close()
    logger.info(f"Student training finished! Checkpoints saved in {save_dir}")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 2: 训练学生模型（FGL 蒸馏 - 动态权重版）")
    logger.info(f"  教师 checkpoint: {Config.teacher_checkpoint}")
    logger.info(f"  初始引导损失权重 λ = {Config.lam}")
    logger.info("=" * 60)
    train_student()
    send_email("学生模型 FGL 蒸馏训练完成！请检查 Checkpoints。")
