import hickle as hkl
import numpy as np

from iono.config import Config

def check_null_values(file_path):
    print(f"Loading data from {file_path}...")
    try:
        data_dict = hkl.load(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    days_data = data_dict.get('data', [])
    if not days_data:
        print("No data found in the 'data' key.")
        return

    total_days = len(days_data)
    print(f"Total days loaded: {total_days}")

    has_nulls = False
    
    for day_idx, day_info in enumerate(days_data):
        doy = day_info.get('doy', 'Unknown')
        
        arrays_to_check = {
            'tec_array': day_info.get('tec_array'),
            'kp_array': day_info.get('kp_array'),
            'dst_array': day_info.get('dst_array'),
            'f107_array': day_info.get('f107_array')
        }

        for name, arr in arrays_to_check.items():
            if arr is None:
                print(f"[Warning] Day {doy}: {name} is None (missing).")
                has_nulls = True
                continue
                
            null_count = np.isnan(arr).sum()
            if null_count > 0:
                print(f"[Alert] Day {doy}: {name} contains {null_count} null (NaN) values.")
                has_nulls = True

    if not has_nulls:
        # success变为绿色
        print("\033[92mSuccess: No null or NaN values found in any of the arrays (tec, kp, dst, f107).\033[0m")
    else:
        print("Finished checking, null values were detected as listed above.")

if __name__ == "__main__":
    for year in range(2016, 2026):
        check_null_values(Config.data_dir / "hickle" / f"gim_{year}_hourlyaux.hickle")
