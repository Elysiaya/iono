"""
消融实验快速使用指南
快速开始消融实验的各个阶段
"""

# ==================== 快速使用 ====================

# 1. 一键运行完整消融实验流程
# =====================================
# cd /path/to/iono
# python ablation_study/run_ablation_study.py

# 运行特定阶段：
# - 只运行训练: python ablation_study/run_ablation_study.py --phase train
# - 只运行评估: python ablation_study/run_ablation_study.py --phase eval
# - 只运行可视化: python ablation_study/run_ablation_study.py --phase viz


# 2. 分阶段运行
# =====================================

# ---- 阶段 1: 训练所有模型 ----
# 训练所有四种模型变体（可以使用主控制脚本）
# python ablation_study/run_ablation_study.py --phase train

# 训练单个模型
# python ablation_study/train_baseline.py
# python ablation_study/train_no_fgl.py
# python ablation_study/train_no_film.py
# 完整模型使用根目录 train_teacher.py 和 train_student.py

# ---- 阶段 2: 评估已训练的模型 ----
# 评估所有模型在验证集上的性能
# python ablation_study/eval_ablation.py

# 指定检查点目录
# python ablation_study/eval_ablation.py --baseline outputs/ablation/checkpoints/.../best_baseline.pth --full outputs/checkpoints/best_student.pth


# ---- 阶段 3: 可视化对比分析 ----
# 生成对比分析图表
# python ablation_study/visualize_ablation.py


# 3. Python 脚本调用
# =====================================

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ---- 直接在 Python 中训练 ----
# 由于拆分，可以通过 subprocess 调用或者直接导入各自的 train 函数
# 例如训练 Baseline 模型:
# from ablation_study.train_baseline import train as train_baseline
# train_baseline()

# ---- 直接在 Python 中评估 ----
from ablation_study.eval_ablation import AblationEvaluator
from ablation_study.ablation_config import EvalConfig

evaluator = AblationEvaluator(config=EvalConfig)
results = evaluator.evaluate_ablation_study(str(EvalConfig.ablation_dir))
evaluator.save_results(str(EvalConfig.results_dir))

# ---- 直接在 Python 中可视化 ----
from ablation_study.visualize_ablation import AblationVisualizer

visualizer = AblationVisualizer(config=EvalConfig)
visualizer.plot_rmse_comparison(results, 'rmse.png')
visualizer.plot_mae_comparison(results, 'mae.png')
visualizer.plot_stepwise_rmse(results, 'stepwise.png')
visualizer.plot_contribution_analysis(results, 'contribution.png')


# 4. 输出说明
# =====================================

# 训练完成后的输出结构：
# outputs/ablation/
# ├── checkpoints/
# │   ├── baseline_20260518_120000/
# │   │   ├── best_baseline.pth              # 最佳基线模型
# │   │   ├── baseline_epoch01_...pth
# │   │   └── baseline_epoch02_...pth
# │   ├── no_fgl_20260518_120000/
# │   ├── no_film_20260518_120000/
# │   └── full_20260518_120000/
# ├── logs/                                  # TensorBoard 日志
# │   ├── baseline/
# │   ├── no_fgl/
# │   ├── no_film/
# │   └── full/
# └── results/
#     ├── ablation_results_20260518_120000.csv       # CSV 格式结果
#     ├── ablation_results_20260518_120000.pkl       # Python pickle 格式
#     ├── ablation_report_20260518_120000.txt        # 详细文本报告
#     ├── rmse_comparison_20260518_120000.png        # RMSE 对比图
#     ├── mae_comparison_20260518_120000.png         # MAE 对比图
#     ├── stepwise_rmse_20260518_120000.png          # 逐步长 RMSE 曲线
#     └── contribution_analysis_20260518_120000.png  # 贡献度分析图


# 5. 监控训练进度
# =====================================

# 实时查看 TensorBoard
# tensorboard --logdir outputs/ablation/logs

# 在浏览器中打开：http://localhost:6006


# 6. 关键文件说明
# =====================================

# ablation_models.py
#   - BaselineModel: w/o Both，基础 ConvLSTM
#   - NoFGLModel: w/o FGL，只有 FiLM
#   - NoFiLMModel: w/o FiLM，只有 FGL
#   - FullModel: w/ Both，完整模型

# ablation_config.py
#   - AblationConfig: 基础配置（数据路径、模型参数）
#   - TrainConfig: 训练特定配置
#   - EvalConfig: 评估特定配置

# train_*.py 系列脚本 (train_baseline.py, train_no_fgl.py, train_no_film.py)
#   - 单独配置并训练四种不同的消融变体模型
#   - train(): 独立的训练入口

# eval_ablation.py
#   - AblationEvaluator: 评估器类
#   - evaluate_ablation_study(): 完整评估流程
#   - compute_metrics(): 计算各项指标
#   - filter_by_dst(): 按 Dst 指数筛选数据

# visualize_ablation.py
#   - AblationVisualizer: 可视化类
#   - plot_rmse_comparison(): RMSE 对比
#   - plot_mae_comparison(): MAE 对比
#   - plot_stepwise_rmse(): 逐步长曲线
#   - plot_contribution_analysis(): 贡献度分析


# 7. 常用命令速查
# =====================================

"""
# 快速开始（推荐）
cd /path/to/iono
python ablation_study/run_ablation_study.py

# 单独训练基线模型
python ablation_study/train_baseline.py

# 评估（假设已完成训练）
python ablation_study/eval_ablation.py

# 只生成可视化
python ablation_study/run_ablation_study.py --phase viz

# 使用自定义路径
python ablation_study/run_ablation_study.py \\
    --checkpoint-dir /custom/path/checkpoints \\
    --output-dir /custom/path/results

# 跳过训练，只进行评估和可视化
python ablation_study/run_ablation_study.py --skip-train
"""


# 8. 预期结果概览
# =====================================

"""
消融实验预期发现：

1. FiLM 的效果（w/o FGL vs w/o Both）
   - 在平静期：改进较小（如 0.5% - 2%）
   - 在磁暴期：改进明显（如 5% - 15%）
   
2. FGL 的效果（w/o FiLM vs w/o Both）
   - 在平静期：稳定改进（如 8% - 12%）
   - 在磁暴期：显著改进（如 10% - 20%）
   
3. 协同效应（w/ Both vs 单独 FGL/FiLM）
   - 总体改进 > FGL 改进 + FiLM 改进
   - 表现为负协同系数（正向协同）
   
4. 长期预报鲁棒性
   - FGL + FiLM 组合在 18-24 小时预报中保持稳定性最佳
   - w/o Both 在长期预报中误差快速增长
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n要运行消融实验，请执行以下命令之一：\n")
    print("1. 完整流程（训练+评估+可视化）:")
    print("   python ablation_study/run_ablation_study.py\n")
    print("2. 只训练某个模型:")
    print("   python ablation_study/train_baseline.py\n")
    print("3. 只评估:")
    print("   python ablation_study/eval_ablation.py\n")
    print("4. 只可视化:")
    print("   python ablation_study/visualize_ablation.py\n")
