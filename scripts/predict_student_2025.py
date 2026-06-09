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

DEFAULT_OUTPUT_NAME = f"student_predictions_2025_{Config.pred_steps}h.npz"


def predict_2025(checkpoint_path=None, save_path=None, data_paths=None):
    Config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path_24 = str(Config.data_dir / "hickle" / "gim_2024_hourlyaux.hickle")
    data_path_25 = str(Config.data_dir / "hickle" / "gim_2025_hourlyaux.hickle")
    data_paths = data_paths or [data_path_24, data_path_25]

    checkpoint_path = checkpoint_path or Config.student_checkpoint
    save_path = save_path or str(Config.results_dir / DEFAULT_OUTPUT_NAME)
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading data from 2024 and 2025...")
    full_dataset = IonosphereDatasetFGL(
        data_paths,
        window_size=Config.window_size,
        future_size=Config.pred_steps,
        pred_steps=Config.pred_steps,
        transform=fgl_normalize_transform,
        return_time=True,
    )

    target_indices = []
    for i in range(len(full_dataset)):
        hist_end = full_dataset.valid_indices[i] + Config.window_size
        target_time = full_dataset.all_times[hist_end]
        if target_time.startswith("2025"):
            target_indices.append(i)

    target_indices = target_indices[::24]
    dataset = Subset(full_dataset, target_indices)
    print(f"Selected {len(dataset)} sequence samples in 2025 for non-overlapping evaluation.")

    loader = DataLoader(
        dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Starting prediction...")
    all_preds = []
    all_trues = []
    all_times = []

    TEC_MAX = 100.0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in tqdm(loader, desc="Predicting 2025"):
            X_hist, aux_hist, X_future, aux_future, y, target_times = batch

            X_hist, aux_hist = X_hist.to(device), aux_hist.to(device)
            aux_future = aux_future.to(device)

            pred_y, _ = model(
                X_hist,
                aux_x=aux_hist,
                future_aux=aux_future,
                dec_aux=aux_future[:, :Config.pred_steps, :],
                return_hidden=True,
                y_true=None,
                tf_ratio=0.0,
            )

            pred_y = pred_y.cpu().float().numpy() * TEC_MAX
            y_true = y.cpu().float().numpy() * TEC_MAX

            all_preds.append(pred_y)
            all_trues.append(y_true)
            all_times.extend(target_times)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    print(f"Prediction done. Shape: {all_preds.shape}")
    print(f"Saving to {save_path} ...")
    np.savez_compressed(
        save_path,
        predictions=all_preds,
        truths=all_trues,
        times=np.array(all_times),
    )

    rmse = np.sqrt(np.mean((all_preds - all_trues) ** 2))
    mae = np.mean(np.abs(all_preds - all_trues))
    print(f"Overall 2025 RMSE: {rmse:.4f}")
    print(f"Overall 2025 MAE: {mae:.4f}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Predict 2025 TEC maps with the trained student model.")
    parser.add_argument("--checkpoint", default=Config.student_checkpoint, help="Path to the student checkpoint.")
    parser.add_argument("--output", default=str(Config.results_dir / DEFAULT_OUTPUT_NAME), help="Output .npz path.")
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
