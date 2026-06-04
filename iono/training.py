from torch.utils.data import DataLoader, Subset

from data_pipeline.fgl_normalize_transform import fgl_normalize_transform
from iono.config import Config
from iono.dataset_fgl import IonosphereDatasetFGL


def _loader_kwargs(shuffle):
    kwargs = {
        "batch_size": Config.batch_size,
        "shuffle": shuffle,
        "num_workers": Config.num_workers,
        "pin_memory": Config.pin_memory,
    }
    if Config.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_temporal_dataloaders(
    hickle_paths,
    window_size,
    future_size,
    pred_steps,
    batch_size,
    logger=None,
):
    dataset = IonosphereDatasetFGL(
        hickle_paths,
        window_size=window_size,
        future_size=future_size,
        pred_steps=pred_steps,
        transform=fgl_normalize_transform,
    )
    total = len(dataset)
    split = int(0.9 * total)
    gap = window_size + future_size

    train_end = max(0, split - gap)
    train_indices = list(range(train_end))
    val_indices = list(range(split, total))

    if not train_indices or not val_indices:
        raise ValueError(
            "Dataset is too small for temporal train/validation split with "
            f"gap={gap}. total={total}, split={split}"
        )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        **{**_loader_kwargs(shuffle=True), "batch_size": batch_size},
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        **{**_loader_kwargs(shuffle=False), "batch_size": batch_size},
    )

    if logger:
        logger.info(
            "Total samples: %s, Train: %s, Val: %s, Temporal gap: %s",
            total,
            len(train_indices),
            len(val_indices),
            gap,
        )
    return train_loader, val_loader
