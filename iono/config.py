import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Config:
    data_dir = Path(os.getenv("IONO_DATA_DIR", PROJECT_ROOT / "data"))
    outputs_dir = Path(os.getenv("IONO_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    results_dir = outputs_dir / "results"

    # ==================== 数据集路径 ====================
    # 优化 1: 使用列表推导式，代码更紧凑
    hickle_paths = [
        str(data_dir / "hickle" / f"gim_{year}_hourlyaux.hickle")
        for year in range(2023, 2026)
    ]
    
    # ==================== 时序序列参数 ====================
    window_size = 72      # 历史输入步长 (3天)
    future_size = 72      # 特权未来步长 (3天) - 注意: 特权输入比预测步长长，有助于全局上下文
    pred_steps = 24       # 预测未来步长 (1天)
    
    # ==================== 模型架构参数 ====================
    in_channels = 3       # 引入位置编码后通道数变为 3 (TEC, lat, lon)
    hidden_channels = 48  
    num_layers = 2
    
    # FiLM 相关参数
    priv_gru_hidden = 32  # GRU 处理 aux_future 的隐藏维度
    
    # ==================== 训练控制参数 ====================
    batch_size = 16       # 如果 OOM，可以减小到 8，并配合 gradient_accumulation_steps=2
    learning_rate = 5e-4
    num_epochs = 50
    
    # 优化 3: 把 DataLoader 和 Scheduler 的参数收拢到 Config
    num_workers = 4
    pin_memory = True
    early_stop_patience = 10
    lr_decay_factor = 0.5
    lr_decay_patience = 4
    
    # ==================== 知识蒸馏 (FGL) 参数 ====================
    # 学生模型模仿教师隐状态的损失权重
    # 提示: 实际训练时需观察 TaskLoss 和 MimicLoss 的初始量级，若 MimicLoss 过大则调小 lam
    lam = 0.5             
    
    # ==================== 计划采样 (Scheduled Sampling) ====================
    tf_start_ratio = 1.0  # 初始 teacher forcing 比例
    tf_end_ratio = 0.0    # 最终 teacher forcing 比例
    tf_decay_epochs = 40  # 线性衰减经过的 epoch 数
    tf_decay_mode = 'linear' # 可选: 'linear', 'exponential', 'inverse_sigmoid'

    # ==================== 模型保存路径 ====================
    # 优化 4: 建立统一的 checkpoints 目录
    teacher_checkpoint = str(checkpoints_dir / "best_teacher.pth")
    student_checkpoint = str(checkpoints_dir / "best_student.pth")

    tensorboard_log_dir = str(logs_dir)  # TensorBoard 日志目录

    resume_ckpt_student = None  # 训练恢复路径，默认为 None 表示不恢复
    
    # 保证输出目录存在
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
