import pandas as pd

from iono.config import Config

def main():
    aux_dir = Config.data_dir / "auxdata"
    input_file = aux_dir / "omni2_c0vmweP_9N.lst"
    output_file = aux_dir / "omni2_c0vmweP_9N.csv"

    # Column names extracted from the .fmt file
    columns = [
        "YEAR", 
        "DOY", 
        "Hour", 
        "Kp index", 
        "Dst-index, nT", 
        "ap_index, nT", 
        "f10.7_index", 
        "AE-index, nT"
    ]

    try:
        # Read the space-separated data
        df = pd.read_csv(input_file, sep=r'\s+', names=columns)
        
        # Save to CSV format
        df.to_csv(output_file, index=False)
        print(f"成功将文件转换并保存为: {output_file}")
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    main()
