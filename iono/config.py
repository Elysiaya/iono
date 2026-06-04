from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.getenv("IONO_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUTS_DIR = Path(os.getenv("IONO_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

class Config:
    data_dir = DEFAULT_DATA_DIR
    outputs_dir = DEFAULT_OUTPUTS_DIR
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    results_dir = outputs_dir / "results"

    # ==================== 数据集路径 ====================
    # 优化 1: 使用列表推导式，代码更紧凑
    hickle_paths = [
        str(DEFAULT_DATA_DIR / "hickle" / f"gim_{year}_hourlyaux.hickle")
        for year in range(2023, 2026)
    ]
    
    # ==================== 时序序列参数 ====================
    window_size = 72      # 历史输入步长 (3天)
    future_size = 72      # 特权未来步长 (3天)
    pred_steps = 6        # 预测未来步长 (6小时)
    
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
    @classmethod
    def ensure_output_dirs(cls):
        for path in (cls.checkpoints_dir, cls.logs_dir, cls.results_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def training_snapshot(cls):
        """Return JSON-friendly training settings stored inside checkpoints."""
        return {
            "paths": {
                "data_dir": str(cls.data_dir),
                "outputs_dir": str(cls.outputs_dir),
                "checkpoints_dir": str(cls.checkpoints_dir),
                "logs_dir": str(cls.logs_dir),
                "results_dir": str(cls.results_dir),
                "hickle_paths": list(cls.hickle_paths),
                "teacher_checkpoint": cls.teacher_checkpoint,
                "student_checkpoint": cls.student_checkpoint,
            },
            "sequence": {
                "window_size": cls.window_size,
                "future_size": cls.future_size,
                "pred_steps": cls.pred_steps,
            },
            "model": {
                "in_channels": cls.in_channels,
                "hidden_channels": cls.hidden_channels,
                "num_layers": cls.num_layers,
                "priv_gru_hidden": cls.priv_gru_hidden,
                "num_aux": 5,
            },
            "training": {
                "batch_size": cls.batch_size,
                "learning_rate": cls.learning_rate,
                "num_epochs": cls.num_epochs,
                "num_workers": cls.num_workers,
                "pin_memory": cls.pin_memory,
                "early_stop_patience": cls.early_stop_patience,
                "lr_decay_factor": cls.lr_decay_factor,
                "lr_decay_patience": cls.lr_decay_patience,
            },
            "fgl": {
                "lam": cls.lam,
                "teacher_uses_future_tec": True,
                "student_uses_future_aux": True,
                "future_aux_features": ["Kp", "Dst", "F10.7", "doy_sin", "doy_cos"],
                "note": "Teacher is a privileged model allowed to see target-window future TEC for distillation; its validation RMSE is not a fair deployment metric.",
            },
            "scheduled_sampling": {
                "tf_start_ratio": cls.tf_start_ratio,
                "tf_end_ratio": cls.tf_end_ratio,
                "tf_decay_epochs": cls.tf_decay_epochs,
                "tf_decay_mode": cls.tf_decay_mode,
            },
        }
