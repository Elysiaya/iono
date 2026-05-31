import os
import numpy as np

def inspect_predictions(file_path):
    print(f"Inspecting file: {file_path}")
    if not os.path.exists(file_path):
        print("File does not exist!")
        return

    # 加载 .npz 文件
    data = np.load(file_path)
    
    # 打印文件中包含的所有键（变量名）
    print(f"\nKeys in file: {data.files}")
    
    # 提取各个变量并打印它们的形状和数据类型
    for key in data.files:
        array = data[key]
        print(f"\n--- {key} ---")
        print(f"Shape: {array.shape}")
        print(f"Type:  {array.dtype}")
        
    # 选取其中的 predictions 和 truths
    preds = data['predictions']
    truths = data['truths']
    times = data['times']
    
    # 打印一些基本的统计信息
    print("\n--- Basic Statistics ---")
    print(f"Predictions - Min: {np.min(preds):.2f}, Max: {np.max(preds):.2f}, Mean: {np.mean(preds):.2f}")
    print(f"Truths      - Min: {np.min(truths):.2f}, Max: {np.max(truths):.2f}, Mean: {np.mean(truths):.2f}")
    
    # 打印前几个时间戳看看
    print("\n--- First 5 Time Stamps ---")
    for t in times[:5]:
        print(t)
        
    # 打印最后几个时间戳看看
    print("\n--- Last 5 Time Stamps ---")
    for t in times[-5:]:
        print(t)

if __name__ == "__main__":
    # 构造文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "results", "student_predictions_2025.npz")
    
    inspect_predictions(file_path)
