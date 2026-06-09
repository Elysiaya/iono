"""
共享 DataLoader。
IonosphereDatasetFGL 返回完整五元组：
    (X_hist, aux_hist, X_future, aux_future, y)
教师直接使用全部字段；学生训练时忽略 X_future（特权信息）即可。
"""

from torch.utils.data import DataLoader, Subset

from data_pipeline.fgl_normalize_transform import fgl_normalize_transform
from iono.config import Config
from iono.dataset_fgl import IonosphereDatasetFGL


def build_dataloaders(
    hickle_paths,
    window_size,
    future_size,
    pred_steps,
    batch_size,
    logger=None,
):
    """
    Returns:
        train_loader:  (X_hist, aux_hist, X_future, aux_future, y)
        val_loader:    (X_hist, aux_hist, X_future, aux_future, y)
    """
    dataset = IonosphereDatasetFGL(
        hickle_paths,
        window_size=window_size,
        future_size=future_size,
        pred_steps=pred_steps,
        transform=fgl_normalize_transform,
    )
    total = len(dataset)
    split = int(0.9 * total)
    train_indices = list(range(split))
    val_indices = list(range(split, total))

    if logger:
        logger.info(
            "Total samples: %s, Train: %s, Val: %s",
            total, len(train_indices), len(val_indices),
        )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=Config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=Config.pin_memory,
    )

    return train_loader, val_loader
