import hickle as hkl
from torch.utils.data import Dataset
import numpy as np
import torch


class IonosphereDatasetFGL(Dataset):
    """
    FGL（未来引导学习）专用数据集。
    返回五元组 (X_hist, aux_hist, X_future, aux_future, y)：
      - X_hist:     (window_size, 1, 71, 73)  历史TEC序列
      - aux_hist:   (window_size, 5)          历史辅助序列 (Kp, Dst, F10.7, doy_sin, doy_cos)
      - X_future:   (future_size, 1, 71, 73)  未来TEC序列（教师特权信息，位于预测窗口之后）
      - aux_future: (future_size, 5)          未来辅助序列
      - y:          (pred_steps, 1, 71, 73)   预测目标（历史窗口后的多步序列）
    """

    def __init__(self, hickle_paths: list, window_size=24, future_size=24, pred_steps=24, transform=None, return_time=False):
        """
        Args:
            hickle_paths: 多个路径的列表(list)
            window_size:  历史窗口大小（小时）
            future_size:  未来窗口大小（小时），教师模型的特权信息长度
            pred_steps:   预测步数（小时）
            return_time:  是否返回当前目标对应的时间戳
        """
        self.window_size = window_size
        self.future_size = future_size
        self.pred_steps = pred_steps
        self.transform = transform
        self.return_time = return_time

        # 逐文件加载，记录每个文件的TEC数据及其在合并数组中的起止索引
        tec_segments = []
        aux_segments = []
        self.all_times = []
        
        import datetime
        import re
        
        for path in hickle_paths:
            data = hkl.load(path)
            
            # 解析年份
            year = data.get('year')
            
            days = data['data']
            file_tec = []
            file_aux = []
            for day in days:
                file_tec.append(day['tec_array'])  # (24, 71, 73)
                
                # 获取当天的年积日 (doy)，并转换为周期性的 sin 和 cos 编码
                # doy 范围通常是 1~365 (闰年 366)，这里简单统一除以 365.25
                doy = day['doy']
                doy_sin = np.sin(2 * np.pi * doy / 365.25)
                doy_cos = np.cos(2 * np.pi * doy / 365.25)
                
                # 为了与每小时的数据对齐，将单天的 sin/cos 扩展为 (24,) 的数组
                doy_sin_arr = np.full_like(day['kp_array'], doy_sin)
                doy_cos_arr = np.full_like(day['kp_array'], doy_cos)
                
                # 组合辅助特征 Kp, Dst, F10.7, doy_sin, doy_cos 为 (24, 5) 矩阵
                aux_matrix = np.stack([
                    day['kp_array'], 
                    day['dst_array'], 
                    day['f107_array'],
                    doy_sin_arr,
                    doy_cos_arr
                ], axis=-1)
                file_aux.append(aux_matrix)
                
                base_time = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(doy) - 1)
                for h in range(len(day['tec_array'])):
                    self.all_times.append((base_time + datetime.timedelta(hours=h)).strftime("%Y-%m-%d %H:00:00"))

            tec_segments.append(np.concatenate(file_tec, axis=0))
            aux_segments.append(np.concatenate(file_aux, axis=0))

        # 记录每段的长度，仅用于可能的信息调试
        self.segment_lengths = [len(seg) for seg in tec_segments]
        self.all_tec = np.concatenate(tec_segments, axis=0)  # (Total_T, 71, 73)
        self.all_aux = np.concatenate(aux_segments, axis=0)  # (Total_T, 3)

        # 直接按顺序拼接所有数据，忽略跨年或缺失数据造成的时间跳变
        total_needed = window_size + future_size  # 历史 + 未来特权
        self.valid_indices = list(range(len(self.all_tec) - total_needed + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]

        # 历史窗口：[start, start + window_size)
        hist_end = start + self.window_size
        X_hist = self.all_tec[start: hist_end]          # (window_size, 71, 73)
        aux_hist = self.all_aux[start: hist_end]        # (window_size, 5)

        # 预测目标窗口：[hist_end, hist_end + pred_steps)
        target_end = hist_end + self.pred_steps
        y = self.all_tec[hist_end: target_end]          # (pred_steps, 71, 73)

        # 未来窗口：[target_end, target_end + future_size) — 位于 y 之后，不与 y 重叠
        future_start = hist_end
        X_future = self.all_tec[future_start: future_start + self.future_size]  # (future_size, 71, 73)
        aux_future = self.all_aux[future_start: future_start + self.future_size]# (future_size, 5)

        # 转为 tensor + 添加 channel 维度 (TEC增加通道维度，辅助数据保持原状)
        X_hist = torch.from_numpy(X_hist).float().unsqueeze(1)      # (24, 1, 71, 73)
        aux_hist = torch.from_numpy(aux_hist).float()               # (24, 5)
        
        X_future = torch.from_numpy(X_future).float().unsqueeze(1)  # (24, 1, 71, 73)
        aux_future = torch.from_numpy(aux_future).float()           # (24, 5)
        
        y = torch.from_numpy(y).float().unsqueeze(1)                # (pred_steps, 1, 71, 73)

        if self.transform:
            X_hist, aux_hist, X_future, aux_future, y = self.transform(
                X_hist, aux_hist, X_future, aux_future, y
            )

        if self.return_time:
            target_time = self.all_times[hist_end]
            return X_hist, aux_hist, X_future, aux_future, y, target_time

        return X_hist, aux_hist, X_future, aux_future, y


if __name__ == "__main__":
    # 提取测试
    dataset_single = IonosphereDatasetFGL(
        ["hickle/gim_2024_hourlyaux.hickle", "hickle/gim_2025_hourlyaux.hickle"],
        return_time=True,
    )
    print(f"Single-file dataset size: {len(dataset_single)} samples")
    X_h, aux_h, X_f, aux_f, y, target_time = dataset_single[0]
    print(f"X_hist: {X_h.shape}, aux_hist: {aux_h.shape}")
    print(f"X_future: {X_f.shape}, aux_future: {aux_f.shape}")
    print(f"y: {y.shape}")
    print(f"Target time start: {target_time}")
    X_h, aux_h, X_f, aux_f, y, target_time = dataset_single[-1]
    print(f"Target time end: {target_time}")
