from datetime import date, datetime, timedelta
import os
from pathlib import Path
import hickle as hkl
import numpy as np

from data_pipeline.read_ionex_file import read_ionex_file
from iono.config import PROJECT_ROOT, Config


DEFAULT_IONEX_ROOT = Path(os.getenv("IONO_IONEX_ROOT", PROJECT_ROOT.parent / "gim"))


def load_omni_data(filepath, target_year):
    """解析 OMNI2 历史文件提取时间、Kp、Dst 和 F10.7 指数值"""
    times_list, kp_list, dst_list, f107_list = [], [], [], []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            # 跳过空行或非数据行
            if not line or line.startswith('YEAR') or line.startswith('FORMAT') or line.startswith('ITEMS'):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                try:
                    year_val = int(parts[0])
                    # 只处理目标年份及其前后的数据
                    if target_year - 1 <= year_val <= target_year + 1:
                        doy_val = int(parts[1])
                        hour_val = int(parts[2])
                        kp_val = float(parts[3])
                        dst_val = float(parts[4])
                        f107_val = float(parts[6])
                        
                        # 根据 OMNI 常见缺失值过滤：Kp=99, Dst=99999, F10.7=999.9
                        if kp_val < 99 and dst_val < 99999 and f107_val < 999.0:
                            # 换算时间戳
                            day_start = datetime(year_val, 1, 1) + timedelta(days=doy_val - 1)
                            dt = day_start + timedelta(hours=hour_val)
                            
                            times_list.append(dt.timestamp())
                            kp_list.append(kp_val)
                            dst_list.append(dst_val)
                            f107_list.append(f107_val)
                except:
                    continue
                    
        times_array = np.array(times_list)
        kp_array = np.array(kp_list)
        dst_array = np.array(dst_list)
        f107_array = np.array(f107_list)
        
        sort_idx = np.argsort(times_array)
        return times_array[sort_idx], kp_array[sort_idx], dst_array[sort_idx], f107_array[sort_idx]
    except Exception as e:
        print(f"Loading OMNI data failed: {e}")
        return None, None, None, None

def main(year, output_dir, omni_file, ionex_root=None):
    output_file = os.path.join(output_dir, f"gim_{year}_hourlyaux.hickle")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ionex_root = Path(ionex_root or DEFAULT_IONEX_ROOT)

    # 1. 提前加载全部分散的有效历史观测点备用
    omni_times, kp_vals, dst_vals, f107_vals = load_omni_data(omni_file, year)
    days = (date(year, 12, 31) - date(year, 1, 1)).days + 1

    all_days = []
    for doy in range(1, days + 1):  # 1 到 365
        if year >=2023:
            filename = f"COD0OPSFIN_{year}{doy:03d}0000_01D_01H_GIM.INX"
        elif year >= 2022 and doy >= 331:
            filename = f"COD0OPSFIN_{year}{doy:03d}0000_01D_01H_GIM.INX"
        else:
            filename = f"codg{doy:03d}0.{str(year)[-2:]}i"
        filepath = ionex_root / f"ionex_{year}" / filename

        tec_array, times = read_ionex_file(filepath)
            
        # IONEX 文件通常包含 00:00 到 24:00 的数据（共 25 帧），其中 24:00 实际上是第二天的 00:00。
        # 为了避免后续数据拼接时出现重复帧，并且统一每天的序列长度为 24，这里去掉第 25 帧。
        if tec_array.shape[0] == 25:
            tec_array = tec_array[:24]
            
        num_frames = tec_array.shape[0]  # 强制统一为 24
        
        # 2. 生成这天中每一帧（每小时）的精确时间戳进行对齐
        day_start = datetime(year, 1, 1) + timedelta(days=doy-1)
        frame_ts = np.array([(day_start + timedelta(hours=i)).timestamp() for i in range(num_frames)])
        
        # 3. 直接通过时间戳匹配对应的小时数据（无需插值）
        # 由于 OMNI 数据本身是 1 小时分辨率，直接查找对应时间即可
        kp_day, dst_day, f107_day = [], [], []
        for ts in frame_ts:
            # 寻找误差在 1 秒内的时间点匹配
            idx = np.where(np.abs(omni_times - ts) < 1.0)[0]
            if len(idx) > 0:
                i = idx[0]
                kp_day.append(kp_vals[i])
                dst_day.append(dst_vals[i])
                f107_day.append(f107_vals[i])
            else:
                # 万一遇到被过滤掉的缺失值，使用NaN作为标识或根据需要处理
                kp_day.append(np.nan)
                dst_day.append(np.nan)
                f107_day.append(np.nan)
                
        kp_array = np.array(kp_day)
        dst_array = np.array(dst_day)
        f107_array = np.array(f107_day)
        
        all_days.append({ 
            "doy": doy, 
            "tec_array": tec_array, 
            "kp_array": kp_array,
            "dst_array": dst_array,
            "f107_array": f107_array,
        })
        # print(f"Processed doy {doy}")
        # print(f"TEC shape: {tec_array.shape}")
        # print(f"Kp shape: {kp_array.shape}")
        # print(f"Dst shape: {dst_array.shape}")
        # print(f"F10.7 shape: {f107_array.shape}")

    hkl.dump({
        "year": year,
        "data": all_days,
        'description': f'Hourly GIM TEC maps & aligned Aux (Kp, Dst, F10.7) ({year}, 1-hour), 24 frames/day, aligned by timestamp, NaN for missing values)'
    }, output_file)


if __name__ == "__main__":
    output_directory = Config.data_dir / "hickle"
    omni_file_path = Config.data_dir / "auxdata" / "omni2_2DF56gWboA.lst"

    for target_year in range(2025, 2026):
        main(target_year, output_directory, omni_file_path)
        print(f"Finished processing year {target_year}, output saved to {output_directory}")                      
