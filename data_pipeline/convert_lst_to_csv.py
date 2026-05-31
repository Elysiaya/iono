import pandas as pd
import os

def main():
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "auxdata", "omni2_c0vmweP_9N.lst")
    output_file = os.path.join(script_dir, "omni2_c0vmweP_9N.csv")

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
