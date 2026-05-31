import torch

def fgl_normalize_transform(X_hist, aux_hist, X_future, aux_future, y):
    """FGL dataset normalization transform (supports multi-step y)."""
    TEC_MAX = 100.0
    
    # 1. TEC 数据归一化 (形状原来为: B, 1, H, W 或者 window_size, 1, H, W)
    X_hist = X_hist / TEC_MAX
    X_future = X_future / TEC_MAX
    y = y / TEC_MAX

    aux_hist = aux_hist.clone()
    aux_future = aux_future.clone()

    # 2. 辅助数据归一化 (Kp: 0~9, Dst: -200~50, F10.7: 70~300)
    aux_hist[:, 0] = aux_hist[:, 0] / 9.0
    aux_future[:, 0] = aux_future[:, 0] / 9.0
    
    aux_hist[:, 1] = (aux_hist[:, 1] + 200.0) / 250.0
    aux_future[:, 1] = (aux_future[:, 1] + 200.0) / 250.0
    
    aux_hist[:, 2] = (aux_hist[:, 2] - 70.0) / 230.0
    aux_future[:, 2] = (aux_future[:, 2] - 70.0) / 230.0
    
    # doy_sin 和 doy_cos 本来就是 -1 到 1，无需再归一化，由网络自身激活函数处理即可

    # 3. 构建时空物理场 (Spatial Positional Encoding)
    # H=71, W=73
    T_hist, _, H, W = X_hist.shape
    lat_grid = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(T_hist, 1, H, W)
    lon_grid = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(T_hist, 1, H, W)
    
    X_hist_spatial = torch.cat([X_hist, lat_grid, lon_grid], dim=1)  # 通道数由 1 变为 3
    
    if X_future is not None:
        T_fut = X_future.shape[0]
        lat_grid_f = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(T_fut, 1, H, W)
        lon_grid_f = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(T_fut, 1, H, W)
        X_future_spatial = torch.cat([X_future, lat_grid_f, lon_grid_f], dim=1)
    else:
        X_future_spatial = X_future

    return X_hist_spatial, aux_hist, X_future_spatial, aux_future, y
