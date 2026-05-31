import numpy as np
import PyIRI
import PyIRI.edp_update as ml
import PyIRI.sh_library as sh
import pandas as pd
import h5py
import os

# ================= 1. 空间网格参数 (经度与纬度) =================
dlon = 5         # 经度空间分辨率（步长），单位：度
dlat = 2.5       # 纬度空间分辨率（步长），单位：度
# 生成二维网格。经度覆盖 [-180, 180]，纬度覆盖 [-87.5, 87.5]
alat_2d, alon_2d = np.mgrid[ -87.5:87.5 + dlat:dlat,-180:180 + dlon:dlon,]
grid_shape = alon_2d.shape  # 应当是 (71, 73)
alon = alon_2d.ravel()  # 展平为一维经度数组
alat = alat_2d.ravel()  # 展平为一维纬度数组

# ================= 2. 时间序列参数 =================
start_date = "2024-01-02"
end_date = "2025-12-30"
dates = pd.date_range(start=start_date, end=end_date, freq='D')

hr_res = 1       # 时间分辨率，单位：小时 (即每隔 1 小时计算一次)
aUT = np.arange(0, 24, hr_res)  # 生成 0 到 23 时的通用时间 (UT) 一维数组

# ================= 3. 模型配置与物理参数 =================
hmF2_model = 'SHU2015'   # F2层最大电子密度对应高度(hmF2)的模型。Also available: 'AMTB2013', 'BSE1979'
foF2_coeff = 'URSI'      # F2层临界频率(foF2)的系数集。Also available: 'CCIR'
coord = 'GEO'            # 坐标系类型：'GEO' 为地理坐标系
                         # Also available: 'QD' for Quasi-Dipole Lon and Quasi-Dipole Lat inputs
                         #                 'MLT' for MLT and Quasi-Dipole Lat inputs

# ================= 4. 太阳与高空间层参数 =================
F107 = 100       # 太阳活动指数 F10.7 (波长10.7厘米的太阳射电通量)
aalt = np.arange(90, 1000, 1)  # 高度数组，从 90km 到 999km，每隔 1km 计算一层

# ================= 5. 输出文件配置 =================
output_file = r"./predict_resultsIRI.h5"

print(f"开始计算 IRI TEC，总天数: {len(dates)}")
print(f"网格形状: {grid_shape}，数据将会保存至 {output_file}")

# ================= 6. 循环计算并写入 HDF5 =================
with h5py.File(output_file, "w") as f:
    # 如果存在IRI这个数据集，则删除重建（避免追加写入）
    if "IRI" in f:
        del f["IRI"]
    dset_tec = f.create_dataset(
        "IRI", 
        shape=(0, grid_shape[0], grid_shape[1]), 
        maxshape=(None, grid_shape[0], grid_shape[1]), 
        dtype=np.float32, 
        compression="gzip"
    )
    timestamps = []

    for i, dt in enumerate(dates):
        year = dt.year
        month = dt.month
        day = dt.day

        print(f"[{i+1}/{len(dates)}] 计算 {year}-{month:02d}-{day:02d} ... ", end="", flush=True)

        try:
            # 1天计算
            F2, F1, E, sun, mag, EDP = sh.IRI_density_1day(
                year, month, day, aUT, alon, alat, aalt, F107,
                coeff_dir=None,
                foF2_coeff=foF2_coeff,
                hmF2_model=hmF2_model,
                coord=coord)

            # 转换为 TEC 并转成二维矩阵 (24, 73, 71)
            TEC_1d = PyIRI.main_library.edp_to_vtec(EDP, aalt, min_alt=0.0, max_alt=202000.0)
            TEC_2d = TEC_1d.reshape(24, grid_shape[0], grid_shape[1])

            # 将这 24 个小时写进 hdf5
            current_len = dset_tec.shape[0]
            dset_tec.resize((current_len + 24, grid_shape[0], grid_shape[1]))
            dset_tec[current_len:] = TEC_2d
            
            for h in range(24):
                ts = f"{year}-{month:02d}-{day:02d} {h:02d}:00:00"
                timestamps.append(ts)
                
            print("完成")
        except Exception as e:
            print(f"错误: {e}")

    # ================= 7. 截取出指定的结束时间戳位置 =================
    target_end = "2025-12-30 22:00:00"
    if target_end in timestamps:
        end_idx = timestamps.index(target_end) + 1
        dset_tec.resize((end_idx, grid_shape[0], grid_shape[1]))
        timestamps = timestamps[:end_idx]
        print(f"已根据目标截断时间，最后一个数据对应时间戳为 {timestamps[-1]}")

    # 将时间戳字符串数组保存进 hdf5 供之后使用
    f.create_dataset("timestamps", data=np.array(timestamps, dtype="S"))
    print(f"\n全部计算保存完毕！共计 {len(timestamps)} 个时间步。")


f = h5py.File('predict_results.h5', 'r')
print(list(f.keys()))
