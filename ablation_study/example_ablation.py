"""
Small examples for the current ablation-study layout.

This file avoids launching training automatically. It is meant as a quick
reference for constructing model variants and for the shell commands used by
the actual experiment scripts.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ablation_study.ablation_config import EvalConfig, TrainConfig
from ablation_study.ablation_models import BaselineModel, NoFGLModel, NoFiLMModel


MODEL_CLASSES = {
    "baseline": BaselineModel,
    "no_fgl": NoFGLModel,
    "no_film": NoFiLMModel,
}


def build_model(name):
    model_cls = MODEL_CLASSES[name]
    return model_cls(
        in_channels=TrainConfig.in_channels,
        hidden_channels=TrainConfig.hidden_channels,
        num_layers=TrainConfig.num_layers,
        num_aux=5,
        pred_steps=TrainConfig.pred_steps,
        priv_gru_hidden=TrainConfig.priv_gru_hidden,
    )


def print_commands():
    print("Train ablation variants:")
    print("  python ablation_study/train_baseline.py")
    print("  python ablation_study/train_no_fgl.py")
    print("  python ablation_study/train_no_film.py")
    print()
    print("Train the full model:")
    print("  python train_teacher.py")
    print("  python train_student.py")
    print()
    print("Evaluate:")
    print("  python ablation_study/eval_ablation.py \\")
    print("    --baseline outputs/ablation/checkpoints/.../best_baseline.pth \\")
    print("    --no_fgl outputs/ablation/checkpoints/.../best_no_fgl.pth \\")
    print("    --no_film outputs/ablation/checkpoints/.../best_no_film.pth \\")
    print("    --full outputs/checkpoints/best_student.pth")
    print()
    print(f"Default ablation checkpoints: {EvalConfig.ablation_dir}")
    print(f"Default ablation results: {EvalConfig.results_dir}")


if __name__ == "__main__":
    print_commands()
