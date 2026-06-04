import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


from iono.dataset_fgl import IonosphereDatasetFGL
from iono.model_fgl import StudentForecaster
from iono.config import Config
from data_pipeline.fgl_normalize_transform import fgl_normalize_transform

def predict_2025(checkpoint_path=None, save_path=None, data_paths=None):
    Config.ensure_output_dirs()
    # ---- 1. 配置路径与超参数 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 指向 2024 和 2025 年的 hickle 数据 (提供 2024 年末的数据作为历史窗口)
    data_path_24 = str(Config.data_dir / "hickle" / "gim_2024_hourlyaux.hickle")
    data_path_25 = str(Config.data_dir / "hickle" / "gim_2025_hourlyaux.hickle")
    data_paths = data_paths or [data_path_24, data_path_25]
    
    # 模型检查点路径 (可以根据需要修改成具体的最新路径)
    checkpoint_path = checkpoint_path or Config.student_checkpoint
    
    save_path = save_path or str(Config.results_dir / "student_predictions_2025.npz")
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    
    # ---- 2. 构建 Dataset 和 DataLoader ----
    print(f"Loading data from 2024 and 2025...")
    full_dataset = IonosphereDatasetFGL(
        data_paths,
        window_size=Config.window_size,
        future_size=Config.pred_steps,  # [修改] 预测时学生模型其实只需要 24 小时的 aux_future，不需要完整的 72 小时
        pred_steps=Config.pred_steps,
        transform=fgl_normalize_transform,
        return_time=True
    )
    
    # 过滤：仅当“预测第一步时间”属于 2025 年才保留
    target_indices = []
    for i in range(len(full_dataset)):
        # 历史窗口末尾是预测的第一步
        hist_end = full_dataset.valid_indices[i] + Config.window_size
        target_time = full_dataset.all_times[hist_end]
        if target_time.startswith("2025"):
            target_indices.append(i)
            
    # 为了避免预测重叠，每天预测一次：以 24小时 为步长提取索引
    target_indices = target_indices[::24]
    
    dataset = Subset(full_dataset, target_indices)
    print(f"Selected {len(dataset)} sequence samples in 2025 for non-overlapping evaluation.")
    
    loader = DataLoader(
        dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # ---- 3. 初始化学生模型并加载权重 ----
    print("Initializing student model...")
    model = StudentForecaster(
        in_channels=Config.in_channels,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        num_aux=5,
        pred_steps=Config.pred_steps,
        priv_gru_hidden=Config.priv_gru_hidden,
    ).to(device)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    # 注意有些时候模型是用 nn.DataParallel 保存的
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    # 处理 DataParallel 保存的权重前缀 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # ---- 4. 开始预测 ----
    print("Starting prediction...")
    all_preds = []
    all_trues = []
    all_times = []
    
    TEC_MAX = 100.0  # 反归一化系数
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in tqdm(loader, desc="Predicting 2025"):
            X_hist, aux_hist, X_future, aux_future, y, target_times = batch
            
            X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
            aux_future = aux_future.to(device)
            
            # 由于是前向预测，未来特权信息(TEC)不可用，但可以在解码阶段传入对应的 auxiliary 数据(如果业务中可用)
            # tf_ratio = 0 表示完全依靠模型自回归，不用 Teacher Forcing
            pred_y, _ = model(
                X_hist, 
                aux_x=aux_hist,
                future_aux=aux_future,
                dec_aux=aux_future[:, :Config.pred_steps, :],
                return_hidden=True,
                y_true=None, 
                tf_ratio=0.0 
            )
            
            # 反归一化
            pred_y = pred_y.cpu().float().numpy() * TEC_MAX
            y_true = y.cpu().float().numpy() * TEC_MAX
            
            all_preds.append(pred_y)
            all_trues.append(y_true)
            all_times.extend(target_times)

    # ---- 5. 拼接并保存结果 ----
    all_preds = np.concatenate(all_preds, axis=0) # (N, pred_steps, 1, 71, 73)
    all_trues = np.concatenate(all_trues, axis=0)
    
    print(f"Prediction done. Shape: {all_preds.shape}")
    print(f"Saving to {save_path} ...")
    
    # 由于数据可能较大，使用 np.savez_compressed
    np.savez_compressed(
        save_path, 
        predictions=all_preds, 
        truths=all_trues, 
        times=np.array(all_times)
    )
    
    # 顺便计算一下 RMSE 展示
    rmse = np.sqrt(np.mean((all_preds - all_trues) ** 2))
    mae = np.mean(np.abs(all_preds - all_trues))
    print(f"Overall 2025 RMSE: {rmse:.4f}")
    print(f"Overall 2025 MAE: {mae:.4f}")
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Predict 2025 TEC maps with the trained student model.")
    parser.add_argument("--checkpoint", default=Config.student_checkpoint, help="Path to the student checkpoint.")
    parser.add_argument("--output", default=str(Config.results_dir / "student_predictions_2025.npz"), help="Output .npz path.")
    parser.add_argument("--data-2024", default=str(Config.data_dir / "hickle" / "gim_2024_hourlyaux.hickle"))
    parser.add_argument("--data-2025", default=str(Config.data_dir / "hickle" / "gim_2025_hourlyaux.hickle"))
    args = parser.parse_args()

    predict_2025(
        checkpoint_path=args.checkpoint,
        save_path=args.output,
        data_paths=[args.data_2024, args.data_2025],
    )


if __name__ == "__main__":
    main()
