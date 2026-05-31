"""
NoFiLM 模型 (w/o FiLM, only FGL) 训练脚本
"""

import os
import sys
import math
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from iono.dataset_fgl import IonosphereDatasetFGL
from iono.config import Config as OriginalConfig
from data_pipeline.fgl_normalize_transform import fgl_normalize_transform
from ablation_study.ablation_models import NoFiLMModel
from ablation_study.ablation_config import TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation_train_nofilm.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def build_dataloaders(config):
    dataset = IonosphereDatasetFGL(
        config.hickle_paths,
        window_size=config.window_size,
        future_size=config.future_size,
        pred_steps=config.pred_steps,
        transform=fgl_normalize_transform,
    )
    total = len(dataset)
    train_size = int(0.9 * total)

    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, total))

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    logger.info(f"NoFiLM samples: {total}, Train: {train_size}, Val: {total - train_size}")
    return train_loader, val_loader


def train_epoch(model, teacher_model, train_loader, device, optimizer, scaler, criterion_pred, criterion_guide, epoch, config):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    train_pred_loss = 0.0
    train_guide_loss = 0.0
    train_total_loss = 0.0
    valid_batches = 0

    if epoch < config.tf_decay_epochs:
        tf_ratio = config.tf_start_ratio - (config.tf_start_ratio - config.tf_end_ratio) * (epoch / config.tf_decay_epochs)
    else:
        tf_ratio = config.tf_end_ratio
    tf_ratio = max(0.3, tf_ratio)

    time_weights = torch.linspace(1.5, 0.5, steps=config.pred_steps, device=device)

    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.num_epochs}] Train")
    for X_hist, aux_hist, X_future, aux_future, y in train_pbar:
        X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
        X_future, aux_future = X_future.to(device), aux_future.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            pred, H_S = model(
                X_hist,
                aux_x=aux_hist,
                dec_aux=aux_future[:, :config.pred_steps, :],
                return_hidden=True,
                y_true=y,
                tf_ratio=tf_ratio
            )

            mse_per_step = torch.mean((pred - y) ** 2, dim=(0, 2, 3))
            L_pred = torch.mean(mse_per_step * time_weights)

            L_total = L_pred
            L_guide_val = 0.0
            
            if teacher_model is not None:
                with torch.no_grad():
                    _, H_T = teacher_model(X_hist, aux_x=aux_hist, future_tec=X_future, future_aux=aux_future, return_hidden=True)
                
                L_guide_ts = []
                for t in range(len(H_S)):
                    h_s_t = H_S[t]
                    h_t_t = H_T[t].detach()
                    if h_s_t.shape[-2:] != h_t_t.shape[-2:]:
                        h_t_t = nn.functional.interpolate(h_t_t, size=h_s_t.shape[-2:], mode='bilinear', align_corners=False)
                    if h_s_t.shape[1] != h_t_t.shape[1]:
                         min_c = min(h_s_t.shape[1], h_t_t.shape[1])
                         l_g = criterion_guide(h_s_t[:, :min_c], h_t_t[:, :min_c])
                    else:
                         l_g = criterion_guide(h_s_t, h_t_t)
                    L_guide_ts.append(l_g)
                
                L_guide = sum(L_guide_ts) / len(L_guide_ts)
                L_total = L_pred + config.lam * L_guide
                L_guide_val = L_guide.item()

        if math.isnan(L_total.item()) or math.isinf(L_total.item()):
            logger.warning("Found NaN/Inf Loss, skipping this batch!")
            continue

        scaler.scale(L_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        train_pred_loss += L_pred.item()
        train_guide_loss += L_guide_val
        train_total_loss += L_total.item()
        valid_batches += 1
        train_pbar.set_postfix({'TotLoss': f"{L_total.item():.4f}", 'PredRMSE': f"{math.sqrt(L_pred.item()) * 100:.4f}"})

    if valid_batches == 0:
        return None
    return (train_pred_loss / valid_batches, train_guide_loss / valid_batches, train_total_loss / valid_batches)


def validate(model, val_loader, device, criterion_pred, config):
    model.eval()
    val_pred_loss = 0.0
    step_mse = torch.zeros(config.pred_steps, device=device)
    total_samples = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
        for X_hist, aux_hist, _, aux_future, y in val_pbar:
            X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
            aux_future = aux_future.to(device)
            y = y.to(device)
            B = y.size(0)
            total_samples += B

            # Validate without FGL
            pred = model(
                X_hist,
                aux_x=aux_hist,
                dec_aux=aux_future[:, :config.pred_steps, :],
                tf_ratio=0.0
            )

            loss = criterion_pred(pred, y)
            val_pred_loss += loss.item()

            for t in range(config.pred_steps):
                step_mse[t] += nn.functional.mse_loss(pred[:, t], y[:, t]).item() * B

    avg_val_pred = val_pred_loss / len(val_loader)
    step_rmse = torch.sqrt(step_mse / total_samples) * 100.0
    return avg_val_pred, step_rmse


def train(resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = config.ablation_dir / f"no_film_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.tensorboard_log_dir / 'no_film'))

    train_loader, val_loader = build_dataloaders(config)

    # 针对 NoFiLM，包含辅助变量
    num_aux = 5
    model = NoFiLMModel(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        num_aux=num_aux,
        pred_steps=config.pred_steps,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load Teacher model for FGL
    teacher_model = None
    teacher_path = Path(OriginalConfig.teacher_checkpoint)
    if teacher_path.exists():
        from iono.model_fgl import TeacherForecaster
        teacher_model = TeacherForecaster(
            in_channels=OriginalConfig.in_channels,
            hidden_channels=OriginalConfig.hidden_channels,
            num_layers=OriginalConfig.num_layers,
            num_aux=5,
            pred_steps=OriginalConfig.pred_steps,
            priv_gru_hidden=OriginalConfig.priv_gru_hidden
        ).to(device)
        teacher_ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
        state_dict = teacher_ckpt.get('model_state_dict', teacher_ckpt)
        try:
            teacher_model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(f"Teacher strict loading failed. Error: {e}")
            teacher_model.load_state_dict(state_dict, strict=False)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        logger.info("Successfully loaded Teacher model for FGL.")
    else:
        logger.warning(f"Teacher model not found at {teacher_path}. FGL Loss will be 0.")

    criterion_pred = nn.MSELoss()
    criterion_guide = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_decay_factor, 
        patience=config.lr_decay_patience, min_lr=config.lr_decay_min
    )
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    start_epoch = 0
    patience_counter = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model_state = model.module if isinstance(model, nn.DataParallel) else model
        model_state.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_pred_loss', float('inf'))

    logger.info("=" * 80)
    logger.info("Training Ablation Model: NoFiLM (w/ FGL, w/o FiLM)")
    logger.info("=" * 80)

    for epoch in range(start_epoch, config.num_epochs):
        losses = train_epoch(model, teacher_model, train_loader, device, optimizer, scaler, criterion_pred, criterion_guide, epoch, config)
        if losses is None:
            break
        avg_pred, avg_guide, avg_total = losses

        avg_val_pred, step_rmse = validate(model, val_loader, device, criterion_pred, config)
        scheduler.step(avg_val_pred)

        val_rmse = math.sqrt(avg_val_pred) * 100.0
        train_rmse = math.sqrt(avg_pred) * 100.0

        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_pred_loss': avg_val_pred,
        }

        name = f"no_film_epoch{epoch+1:02d}_valRMSE{val_rmse:.4f}.pth"
        torch.save(checkpoint, save_dir / name)

        if avg_val_pred < best_val_loss:
            best_val_loss = avg_val_pred
            torch.save(checkpoint, save_dir / f"best_no_film.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                break

        logger.info(f"[NoFiLM] Epoch [{epoch+1}/{config.num_epochs}] Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Guide Loss: {avg_guide:.4f}")
        writer.add_scalar('loss/train_pred', avg_pred, epoch)
        writer.add_scalar('loss/train_guide', avg_guide, epoch)
        writer.add_scalar('loss/val', avg_val_pred, epoch)
        writer.add_scalar('rmse/train', train_rmse, epoch)
        writer.add_scalar('rmse/val', val_rmse, epoch)

    writer.close()

if __name__ == "__main__":
    train()
