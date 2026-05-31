# 消融实验框架 (Ablation Study Framework)

## 概述

本框架用于验证 FGL（Future Guidance Learning）与 FiLM（Feature-wise Linear Modulation）两大核心创新机制的独立物理贡献与协同增效作用。

## 实验设计

### 模型变体 (Model Variants)

| 模型 | 缩写 | FGL | FiLM | 描述 |
|------|------|-----|------|------|
| 基线模型 | w/o Both | ✗ | ✗ | 纯 ConvLSTM，早期融合辅助特征 |
| 无 FGL | w/o FGL | ✗ | ✓ | 仅有晚期特征调制（FiLM） |
| 无 FiLM | w/o FiLM | ✓ | ✗ | 仅有知识蒸馏（FGL），早期融合 |
| 完整模型 | w/ Both | ✓ | ✓ | FGL + FiLM 完整模型 |

### 评估设置 (Evaluation Setup)

为凸显非平稳环境对模型的真实考验，实验提取了测试集中电离层处于以下两个极端状态的数据子集：

- **长期平静态**: `Dst > -30 nT`
- **极端磁暴态**: `Dst < -100 nT`

分离式量化评估各模型的性能表现。

## 项目结构

```
ablation_study/
├── ablation_models.py          # 四种模型变体定义
├── ablation_config.py          # 统一配置文件
├── train_baseline.py           # Baseline 训练脚本
├── train_no_fgl.py             # w/o FGL 训练脚本
├── train_no_film.py            # w/o FiLM 训练脚本
├── eval_ablation.py            # 评估脚本
├── visualize_ablation.py       # 可视化脚本
├── checkpoints/                # 保存训练的模型检查点
├── logs/                       # TensorBoard 日志
└── results/                    # 评估结果与图表
```

## 快速开始

### 1. 训练所有模型

```bash
cd /path/to/iono
python ablation_study/train_baseline.py
python ablation_study/train_no_fgl.py
python ablation_study/train_no_film.py
```

训练单个模型：

```bash
python ablation_study/train_baseline.py
python ablation_study/train_no_fgl.py
python ablation_study/train_no_film.py
# 完整模型使用根目录 train_teacher.py 和 train_student.py
```

从断点恢复：

```bash
python ablation_study/train_no_fgl.py
```

### 2. 评估所有模型

```bash
python ablation_study/eval_ablation.py \
  --baseline outputs/ablation/checkpoints/.../best_baseline.pth \
  --no_fgl outputs/ablation/checkpoints/.../best_no_fgl.pth \
  --no_film outputs/ablation/checkpoints/.../best_no_film.pth \
  --full outputs/checkpoints/best_student.pth
```

### 3. 生成可视化

```bash
python ablation_study/visualize_ablation.py --results-file results.pkl
```

## 关键指标

### 全局指标 (Global Metrics)

- **RMSE (Root Mean Square Error)**: 整体预测误差
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **Correlation**: 预测与实际的相关系数

### 逐时间步指标 (Step-wise Metrics)

- **Step-wise RMSE**: 不同预报时间的预测准确度
- **Step-wise MAE**: 不同时间步的平均误差

### 贡献度分析

- **FGL 独立贡献**: `Baseline RMSE - (w/o FiLM) RMSE`
- **FiLM 独立贡献**: `Baseline RMSE - (w/o FGL) RMSE`
- **协同效应**: `(w/ Both) RMSE - (w/o FGL) RMSE - (w/o FiLM) RMSE + Baseline RMSE`

## 模型细节

### BaselineModel (w/o Both)

纯粹的时空编码-解码架构：

```python
编码器: ConvLSTM 多层
    ↓
直接解码: ConvLSTM 多层（无特殊调制机制）
    ↓
预测输出
```

特点：
- 辅助特征在输入层直接拼接（早期融合）
- 无知识蒸馏，纯粹独立学习

### NoFGLModel (w/o FGL)

具有 FiLM 晚期特征调制：

```python
编码器: ConvLSTM 多层
    ↓
PrivGRU: 处理未来辅助序列
    ↓
FiLM 调制: γ, β = conv(aux_embedding)
    ↓
调制隐状态: h' = h * (1 + γ) + β
    ↓
解码器: ConvLSTM 多层
    ↓
预测输出
```

特点：
- 辅助特征通过 FiLM 层进行晚期调制
- 无知识蒸馏，直接学习使用辅助信息

### NoFiLMModel (w/o FiLM)

具有 FGL 知识蒸馏但无 FiLM：

```python
编码器: ConvLSTM 多层
    ↓
早期融合: 直接拼接辅助特征
    ↓
解码器: ConvLSTM 多层
    ↓
预测输出 + 隐状态
    ↓
FGL 损失: 模仿教师隐状态
```

特点：
- 辅助特征在输入层拼接（早期融合）
- 有知识蒸馏，模仿教师模型表征

### FullModel (w/ Both)

完整的 FGL + FiLM 模型：

```python
编码器: ConvLSTM 多层
    ↓
PrivGRU + FiLM: 晚期调制
    ↓
调制隐状态: h' = h * (1 + γ) + β
    ↓
解码器: ConvLSTM 多层
    ↓
预测输出 + 隐状态
    ↓
FGL 损失: 模仿教师隐状态
```

特点：
- 晚期特征调制（FiLM）
- 知识蒸馏（FGL）
- 两种机制完整协作

## 配置参数

关键配置参数定义在 `ablation_config.py` 中，保证所有模型训练的一致性：

```python
# 模型架构
hidden_channels = 48
num_layers = 3          # ConvLSTM 层数
priv_gru_hidden = 32    # FiLM GRU 隐藏维度

# 训练超参数
learning_rate = 5e-4
batch_size = 16
num_epochs = 50

# 学习率衰减
lr_decay_factor = 0.5
lr_decay_patience = 6

# 计划采样 (Scheduled Sampling)
tf_start_ratio = 1.0
tf_end_ratio = 0.0
tf_decay_epochs = 40

# FGL 相关
lam = 0.5              # 引导损失权重
alpha_soft = 0.5       # Soft-Target KD 权重
```

## 输出解释

### 训练日志

```
[w/o Both (Baseline)] Epoch [1/50]
  Train RMSE: 12.3456
  Val RMSE: 14.5678 (Best: 14.5678)
  Step 1-24 RMSE: [8.23, 10.45, 12.67, ..., 18.90]
  LR: 5.00e-04
```

### 评估报告

```
w/ Both (Full) (full)
Description: FGL + FiLM 完整模型
Use FGL: True, Use FiLM: True
────────────────────────────────

  Overall:
    RMSE: 12.3456
    MAE: 9.8765
    Correlation: 0.8934
    Step-wise RMSE: 8.23, 10.45, 12.67, ..., 18.90

  Quiet Period:
    RMSE: 10.2345
    MAE: 7.8901
    Correlation: 0.9234
    ...

  Storm Period:
    RMSE: 15.6789
    MAE: 12.3456
    Correlation: 0.7654
    ...
```

## 贡献度分析解释

### 例示结果

假设实验结果如下（单位：TECU）：

| 指标 | Baseline | w/o FGL | w/o FiLM | w/ Both |
|------|----------|---------|----------|---------|
| Overall RMSE | 14.56 | 13.45 | 12.78 | 11.23 |
| Quiet RMSE | 10.23 | 9.87 | 9.45 | 9.12 |
| Storm RMSE | 18.90 | 17.34 | 15.67 | 13.45 |

### 计算贡献度

1. **FiLM 独立贡献**（Overall 期间）:
   ```
   Baseline - w/o FGL = 14.56 - 13.45 = 1.11 TECU ↓
   ```
   FiLM 将 RMSE 降低 1.11 TECU（7.6%）

2. **FGL 独立贡献**（Overall 期间）:
   ```
   Baseline - w/o FiLM = 14.56 - 12.78 = 1.78 TECU ↓
   ```
   FGL 将 RMSE 降低 1.78 TECU（12.2%）

3. **协同效应**（Overall 期间）:
   ```
   (w/ Both) - (w/o FGL) - (w/o FiLM) + Baseline
   = 11.23 - 13.45 - 12.78 + 14.56
   = -0.44 TECU
   ```
   负值表示有正的协同效应，即两种机制协作优于独立应用

## 极端环保期性能分析

在磁暴期（Dst < -100 nT）：

- **w/o Both**: 预报滞后长达 [X] 小时，"迟钝"现象明显
- **w/o FGL** (FiLM): 将磁暴期 RMSE 相比基线压降 [X]%，感知敏锐度显著提升
- **w/o FiLM** (FGL): 相比基线误差峰值削减 [X] TECU，展现高效的抗衰减稳定性
- **w/ Both**: 综合 RMSE 取得全场最优，两机制完美协同

## TensorBoard 可视化

训练期间实时监控各模型的学习过程：

```bash
tensorboard --logdir outputs/ablation/logs
```

监控指标：
- 训练/验证损失
- RMSE 曲线
- 学习率动态
- 时间衰减权重变化

## 常见问题

### Q: 为什么所有模型的超参数要保持一致？

**A**: 这是消融实验的核心原则。只有在所有其他条件一致的情况下，我们才能准确隔离 FGL 和 FiLM 各自的贡献。

### Q: 如何解释负的协同效应？

**A**: 负值表示正协同效应——两种机制协作时的性能提升超过了它们独立应用的改进之和。这说明 FGL 和 FiLM 存在正反馈关系。

### Q: 为什么需要分别评估平静期和磁暴期？

**A**: 电离层在不同空间天气状态下的物理机制不同。分离评估能揭示各机制在不同条件下的鲁棒性。

## 参考文献

- Student Forecaster Architecture: [StudentForecaster](../iono/model_fgl.py)
- Teacher Forecaster Architecture: [TeacherForecaster](../iono/model_fgl.py)
- FGL 知识蒸馏: Hinton et al., 2015
- FiLM 特征调制: Perez et al., 2018

## 许可证

[Your License Here]

## 联系方式

如有问题，请提交 Issue 或联系开发团队。
