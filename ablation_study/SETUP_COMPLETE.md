# Ablation Study Status

The ablation study now uses separate scripts for each variant instead of a
single `train_ablation.py` entrypoint.

## Current Scripts

- `train_baseline.py`: baseline model without FGL or FiLM.
- `train_no_fgl.py`: FiLM-only variant.
- `train_no_film.py`: FGL-only variant.
- `eval_ablation.py`: evaluates ablation checkpoints and the full student model.
- `visualize_ablation.py`: generates comparison plots.
- `run_ablation_study.py`: optional orchestration wrapper.

The full model is still trained from the repository root:

```bash
python train_teacher.py
python train_student.py
```

## Default Outputs

Runtime artifacts are written outside the source directory:

```text
outputs/
├── checkpoints/
├── logs/
├── results/
└── ablation/
    ├── checkpoints/
    ├── logs/
    └── results/
```

You can override the base output directory with `IONO_OUTPUT_DIR`.

## Common Commands

```bash
python ablation_study/train_baseline.py
python ablation_study/train_no_fgl.py
python ablation_study/train_no_film.py
```

```bash
python ablation_study/eval_ablation.py \
  --baseline outputs/ablation/checkpoints/.../best_baseline.pth \
  --no_fgl outputs/ablation/checkpoints/.../best_no_fgl.pth \
  --no_film outputs/ablation/checkpoints/.../best_no_film.pth \
  --full outputs/checkpoints/best_student.pth
```

```bash
tensorboard --logdir outputs/ablation/logs
```
