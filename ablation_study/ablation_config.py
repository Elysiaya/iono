"""
消融实验配置文件
定义消融实验中各个模型变体的超参数和训练配置
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class AblationConfig:
    outputs_root = PROJECT_ROOT / "outputs"
    """消融实验统一配置"""
    
    # ==================== 数据集路径 ====================
    hickle_paths = [
        str(PROJECT_ROOT / "data" / "hickle" / f"gim_{year}_hourlyaux.hickle")
        for year in range(2023, 2026)
    ]
    
    # ==================== 时序序列参数 ====================
    window_size = 72      # 历史输入步长 (3天)
    future_size = 72      # 特权未来步长 (3天)
    pred_steps = 24       # 预测未来步长 (1天)
    
    # ==================== 模型架构参数（保持一致） ====================
    in_channels = 3       # TEC + 位置编码
    hidden_channels = 48  
    num_layers = 2        # ConvLSTM 层数（消融实验中保持与主模型一致）
    priv_gru_hidden = 32  # FiLM 调制的 GRU 隐藏维度
    
    # ==================== 训练超参数（绝对一致） ====================
    batch_size = 16
    learning_rate = 5e-4
    num_epochs = 20
    num_workers = 4
    pin_memory = True
    
    # ==================== 学习率衰减 ====================
    lr_decay_factor = 0.5
    lr_decay_patience = 6
    lr_decay_min = 1e-6
    
    # ==================== Early Stopping ====================
    early_stop_patience = 12
    
    # ==================== 计划采样（Scheduled Sampling） ====================
    tf_start_ratio = 1.0
    tf_end_ratio = 0.0
    tf_decay_epochs = 40
    
    # ==================== FGL 相关参数 ====================
    # 仅对 'no_film' 和 'full' 模型有效
    lam = 0.5             # 引导损失权重
    alpha_soft = 0.5      # Soft-Target KD 权重
    
    # ==================== 梯度裁剪 ====================
    grad_clip_norm = 0.5
    
    # ==================== 输出目录 ====================
    ablation_dir = Path(os.getenv("IONO_ABLATION_CHECKPOINT_DIR", outputs_root / "ablation" / "checkpoints"))
    tensorboard_log_dir = Path(os.getenv("IONO_ABLATION_LOG_DIR", outputs_root / "ablation" / "logs"))
    results_dir = Path(os.getenv("IONO_ABLATION_RESULTS_DIR", outputs_root / "ablation" / "results"))
    
    # 创建输出目录
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ==================== 极端气象子集筛选参数 ====================
    # 用于分离评估稳定期和磁暴期的性能
    QUIET_DST_THRESHOLD = -30    # nT，Dst > -30 为"长期平静态"
    STORM_DST_THRESHOLD = -100   # nT，Dst < -100 为"极端磁暴态"


class TrainConfig(AblationConfig):
    """训练配置"""
    pass


class EvalConfig(AblationConfig):
    """评估配置"""
    pass
