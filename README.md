# FGL-Iono

基于未来引导学习（Future-Guided Learning, FGL）的全球电离层 TEC 多步预报项目。项目采用教师-学生两阶段训练：教师模型在训练时额外利用未来窗口信息，学生模型在推理时只使用历史观测和辅助特征，并通过隐状态模仿、软目标蒸馏等方式学习教师的时空表征。

## 项目概览

当前代码主要面向 24 小时 TEC 预报。默认设置使用过去 72 小时的 TEC 地图作为历史输入，预测未来 24 小时的 TEC 地图，并为教师模型提供额外的未来窗口作为训练阶段的特权信息。

核心特性：

- 教师模型 `TeacherForecaster`：历史 TEC + 未来特权 TEC + 未来辅助特征，通过空间 FiLM 调制历史隐状态。
- 学生模型 `StudentForecaster`：推理阶段不使用未来 TEC，只使用历史 TEC 和可用/预测的辅助特征。
- 数据增强式输入：TEC 通道会拼接经纬度位置编码，模型输入通道为 `3 = TEC + lat + lon`。
- 辅助特征：`Kp`、`Dst`、`F10.7`、`doy_sin`、`doy_cos` 共 5 维。
- 训练策略：AMP 混合精度、梯度裁剪、ReduceLROnPlateau、Early Stopping、Scheduled Sampling。
- 学生蒸馏：预测损失 + 教师隐状态引导损失 + 教师输出软目标损失。

## 目录结构

```text
iono/
├── iono/
│   ├── config.py                 # 全局配置：数据路径、模型参数、训练参数、checkpoint 路径
│   ├── dataset_fgl.py            # FGL 数据集，返回历史窗口、未来窗口、预测目标和时间戳
│   ├── model_fgl.py              # TeacherForecaster / StudentForecaster / ConvLSTM / SpatialFiLM
│   ├── training.py               # 共享 DataLoader 构建与时间序列切分
│   ├── train/
│   │   ├── teacher.py            # Phase 1：训练教师模型
│   │   └── student.py            # Phase 2：训练学生模型
│   └── eval/
│       ├── predict_student_2025.py # 2025 年学生模型批量预测
│       └── inspect_npz.py        # 查看 npz 预测结果
├── data_pipeline/
│   ├── __init__.py
│   ├── read_ionex_file.py        # IONEX 文件解析
│   ├── createhkl.py              # 生成 hickle 数据
│   ├── createhkl_c1pg.py
│   ├── download_igs.py           # 下载 IGS IONEX 数据
│   ├── download_c1pg.py          # 下载 C1PG IONEX 数据
│   ├── fgl_normalize_transform.py# TEC、辅助特征归一化与空间位置编码
│   ├── convert_lst_to_csv.py     # OMNI/辅助数据转换
│   └── check_nulls.py            # 缺失值检查
├── iono_model/
│   ├── IRI.py                    # IRI 经验模型
│   └── Klobuchar.py              # Klobuchar 经验模型
├── scripts/
│   ├── predict_student_2025.py   # 兼容入口，转发到 iono.eval
│   ├── inspect_npz.py            # 兼容入口，转发到 iono.eval
│   ├── test.py                   # 教师模型加载测试
│   └── send_email.py             # 训练完成邮件通知
├── outputs/                      # checkpoint、日志、预测结果（已忽略）
├── train_teacher.py              # 兼容入口，转发到 iono.train.teacher
├── train_student.py              # 兼容入口，转发到 iono.train.student
├── pyproject.toml                # uv 项目配置及依赖
└── README.md
```

## 环境安装

本项目使用 [uv](https://docs.astral.sh/uv/) 作为包管理器。建议使用 Python 3.14+，并优先在 CUDA 环境下训练。

```bash
uv sync
```

该命令将自动读取 `pyproject.toml` 安装相关依赖（如 `torch`、`numpy`、`hickle`、`tensorboard` 等），并配置好含 CUDA 13.0 支持的 PyTorch 环境。如果需要进入虚拟环境：

```bash
uv run python train_teacher.py
# 或激活虚拟环境：
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

## 数据准备

训练脚本默认读取 `iono/config.py` 中的路径：

```python
hickle_paths = [
    data/hickle/gim_2023_hourlyaux.hickle,
    data/hickle/gim_2024_hourlyaux.hickle,
    data/hickle/gim_2025_hourlyaux.hickle,
]
```

每个 hickle 文件应包含按天组织的数据，典型结构如下：

```python
{
    "year": 2025,
    "data": [
        {
            "doy": 1,
            "tec_array": np.ndarray,   # (24, 71, 73)
            "kp_array": np.ndarray,    # (24,)
            "dst_array": np.ndarray,   # (24,)
            "f107_array": np.ndarray,  # (24,)
        },
        ...
    ]
}
```

`IonosphereDatasetFGL` 会把多个年份顺序拼接，并为每个样本返回：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `X_hist` | `(72, 3, 71, 73)` | 历史 TEC + 经纬度位置编码 |
| `aux_hist` | `(72, 5)` | 历史 Kp、Dst、F10.7、doy_sin、doy_cos |
| `X_future` | `(72, 3, 71, 73)` | 教师训练使用的未来窗口 |
| `aux_future` | `(72, 5)` | 未来辅助特征 |
| `y` | `(24, 1, 71, 73)` | 未来 24 小时 TEC 预测目标 |

归一化逻辑在 `data_pipeline/fgl_normalize_transform.py` 中：

- TEC 除以 `100.0`。
- Kp 除以 `9.0`。
- Dst 使用 `(Dst + 200) / 250`。
- F10.7 使用 `(F10.7 - 70) / 230`。
- `doy_sin` 和 `doy_cos` 保持原始周期编码。

## 配置说明

主要参数集中在 `iono/config.py`：

| 参数 | 当前默认值 | 说明 |
| --- | --- | --- |
| `window_size` | `72` | 历史输入窗口，单位小时 |
| `future_size` | `72` | 教师特权未来窗口，单位小时 |
| `pred_steps` | `24` | 预测步长，单位小时 |
| `in_channels` | `3` | TEC + lat + lon |
| `hidden_channels` | `48` | ConvLSTM 隐通道数 |
| `num_layers` | `2` | ConvLSTM 层数 |
| `priv_gru_hidden` | `32` | 未来辅助特征 GRU 隐维度 |
| `batch_size` | `16` | batch size |
| `learning_rate` | `5e-4` | 初始学习率 |
| `num_epochs` | `50` | 最大训练轮数 |
| `num_workers` | `4` | DataLoader 进程数 |
| `early_stop_patience` | `10` | 教师训练 early stopping patience |
| `lam` | `0.5` | 学生隐状态引导损失权重 |
| `tf_start_ratio` | `1.0` | Scheduled Sampling 起始 teacher forcing 比例 |
| `tf_end_ratio` | `0.0` | Scheduled Sampling 结束比例 |
| `tf_decay_epochs` | `40` | teacher forcing 衰减轮数 |
| `teacher_checkpoint` | `outputs/checkpoints/best_teacher.pth` | 学生训练加载的教师权重 |
| `student_checkpoint` | `outputs/checkpoints/best_student.pth` | 默认学生权重路径 |
| `tensorboard_log_dir` | `outputs/logs` | TensorBoard 日志目录 |

可通过环境变量 `IONO_DATA_DIR` 和 `IONO_OUTPUT_DIR` 覆盖默认数据目录和输出目录。

## 训练流程

### 1. 训练教师模型

```bash
uv run python train_teacher.py
```

也可以使用包化入口：

```bash
uv run python -m iono.train.teacher
```

教师模型使用历史窗口、未来 TEC 和未来辅助特征，优化标准 MSE 预测损失。训练过程会：

- 以时间顺序做 9:1 划分，并在训练集和验证集之间留出 `window_size + future_size` 的间隔，避免窗口重叠造成信息泄漏。
- 使用 AMP 混合精度。
- 对 decoder 使用 Scheduled Sampling。
- 保存每轮 checkpoint。
- 在验证集提升时保存 `best_teacher.pth`。

输出目录形如：

```text
checkpoints_teacher_fgl_YYYYMMDD_HHMMSS/
├── best_teacher.pth
└── teacher_epochXX_valRMSE..._trainRMSE....pth
```

教师训练会同时保存时间戳目录下的最佳权重，并更新统一入口：

```text
outputs/checkpoints/best_teacher.pth
```

### 2. 训练学生模型

```bash
uv run python train_student.py
```

也可以使用包化入口：

```bash
uv run python -m iono.train.student
```

学生模型加载 `Config.teacher_checkpoint`，冻结教师模型，并优化：

```text
L_total = L_pred + lam * L_guide + alpha_soft * L_soft
```

其中：

- `L_pred`：带时间权重的预测 MSE，前几个预报小时权重更高。
- `L_guide`：学生隐状态对齐教师隐状态，包含 MSE 和 cosine 距离。
- `L_soft`：学生预测对齐教师预测输出。
- `alpha_soft`：当前脚本中固定为 `0.5`。

学生 checkpoint 默认保存到：

```text
outputs/checkpoints/student_fgl_YYYYMMDD_HHMMSS/
├── best_student.pth
└── student_epochXX_valRMSE..._trainRMSE....pth
```

如需断点续训，在 `iono/config.py` 中设置：

```python
resume_ckpt_student = "outputs/checkpoints/.../student_epochXX_....pth"
```

## 推理与评估

### 预测 2025 年结果

```bash
uv run python scripts/predict_student_2025.py
```

也可以使用包化入口：

```bash
uv run python -m iono.eval.predict_student_2025
```

该脚本会：

- 读取 `data/hickle/gim_2024_hourlyaux.hickle` 和 `data/hickle/gim_2025_hourlyaux.hickle`。
- 使用 2024 年末数据为 2025 年初提供历史窗口。
- 按 24 小时间隔抽样，避免每日预测窗口重叠。
- 默认加载 `Config.student_checkpoint`，也可通过 `--checkpoint` 指定。
- 输出压缩结果到 `outputs/results/student_predictions_2025.npz`，也可通过 `--output` 指定。

保存字段：

```python
predictions  # (N, 24, 1, 71, 73)，已反归一化为 TECU
truths       # (N, 24, 1, 71, 73)，已反归一化为 TECU
times        # 每个样本预测窗口起始时间
```

默认 checkpoint 来自 `Config.student_checkpoint`，也可以通过 `--checkpoint` 指向任意学生模型权重。

### 测试教师权重加载

```bash
uv run python scripts/test.py
```

该脚本会按 `Config.teacher_checkpoint` 实例化并尝试加载教师模型，适合排查模型结构参数和 checkpoint 是否匹配。

### 查看预测结果

```bash
uv run python scripts/inspect_npz.py
```

也可以使用包化入口，或直接指定文件：

```bash
uv run python -m iono.eval.inspect_npz
uv run python -m iono.eval.inspect_npz outputs/results/student_predictions_2025.npz
```

用于快速检查 `npz` 文件中的数组名称、形状和基本内容。

## TensorBoard

训练日志默认写到 `outputs/logs`。例如：

```bash
tensorboard --logdir outputs/logs
```

或在本地修改配置后：

```bash
tensorboard --logdir runs
```

消融实验日志：

```bash
tensorboard --logdir outputs/ablation/logs
```

## 消融实验

消融实验已拆为同级独立项目 `../ablation_study/`，用于比较 FGL 和 FiLM 两种机制在模型中的贡献。

请参阅消融实验项目内的 README 或快速开始脚本。

核心模型变体：

| 变体 | 训练脚本 | 说明 |
| --- | --- | --- |
| `baseline` | `../ablation_study/train_baseline.py` | 不使用 FGL，不使用 FiLM |
| `no_fgl` | `../ablation_study/train_no_fgl.py` | 使用 FiLM，不使用 FGL |
| `no_film` | `../ablation_study/train_no_film.py` | 使用 FGL，不使用 FiLM |
| `full` | `train_student.py` | 完整学生模型，使用 FGL + FiLM |

支持通过主控脚本一键运行流程：
```bash
cd ../ablation_study
uv run python run_ablation_study.py
```
*(注：完整模型依然通过根目录训练)*

评估结果与图表将默认输出至：

```text
outputs/ablation/results/
```

## 经验模型基线

`iono_model/` 下保留了传统电离层经验模型相关代码：

- `IRI.py`
- `Klobuchar.py`

这些脚本可用于与深度学习模型结果做对比分析。

## 常见注意事项

- 数据文件较大，`data/`、`.hickle`、checkpoint 和日志已在 `.gitignore` 中忽略。
- Windows 本地运行时，如果 DataLoader 多进程或 CUDA 初始化有问题，可以先把 `Config.num_workers` 改为 `0` 排查。
- `train_teacher.py` 和 `train_student.py` 会调用 `scripts/send_email.py` 发送训练通知。如果没有配置邮件环境，请按需注释相关调用。
- 若加载 checkpoint 报 shape mismatch，请确认 `Config.in_channels`、`hidden_channels`、`num_layers`、`num_aux=5`、`pred_steps` 与训练该 checkpoint 时完全一致。
- `torch.amp.autocast('cuda')` 需要 CUDA 环境；如果只在 CPU 上运行，可能需要改为禁用 AMP。

## 推荐运行顺序

```bash
# 1. 安装依赖
uv sync

# 2. 确认 data/hickle/ 下存在 2023-2025 hickle 数据

# 3. 训练教师
uv run python train_teacher.py

# 4. 训练学生
uv run python train_student.py

# 5. 预测；默认读取 outputs/checkpoints/best_student.pth
uv run python scripts/predict_student_2025.py

# 6. 可选：运行消融实验
cd ../ablation_study
uv run python train_baseline.py
uv run python train_no_fgl.py
uv run python train_no_film.py
uv run python eval_ablation.py
```
