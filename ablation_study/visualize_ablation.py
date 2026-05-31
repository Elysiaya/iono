"""
消融实验可视化脚本
生成对比分析的图表，直观展示 FGL 和 FiLM 的贡献
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ablation_study.ablation_config import EvalConfig

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationVisualizer:
    """消融实验可视化工具"""
    
    def __init__(self, config=None):
        self.config = config or EvalConfig
        self.colors = {
            'baseline': '#FF6B6B',      # 红色
            'no_fgl': '#4ECDC4',        # 青色
            'no_film': '#45B7D1',       # 蓝色
            'full': '#96CEB4',          # 绿色
        }
        self.model_labels = {
            'baseline': 'Baseline',
            'no_fgl': 'w/o FGL',
            'no_film': 'w/o FiLM',
            'full': 'Student (Ours)',
        }
    
    def plot_rmse_comparison(self, results_dict, save_path=None):
        """绘制 RMSE 对比图表"""
        # 注意: 你的实际 npz 里只存了 'overall_rmse' 和 'per_step_rmse'
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle('RMSE Comparison Across Ablation Models', fontsize=14, fontweight='bold')
        
        models = []
        rmses = []
        colors = []
        
        for model_name, results in results_dict.items():
            if 'overall_rmse' in results:
                models.append(self.model_labels.get(model_name, model_name))
                rmses.append(results['overall_rmse'])
                colors.append(self.colors.get(model_name, 'gray'))
        
        bars = ax.bar(models, rmses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 在柱子上标注数值
        for bar, rmse in zip(bars, rmses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if rmses:
            ax.set_ylim(0, max(rmses) * 1.15)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved RMSE comparison plot to {save_path}")
        plt.show()

    def plot_mae_comparison(self, results_dict, save_path=None):
        """绘制 MAE 对比图表"""
        # npz 中如果只存了 RMSE，这里应该 skip 或处理 KeyError
        logger.warning("MAE data not found in results, skipping plot_mae_comparison")
        pass
    
    def plot_stepwise_rmse(self, results_dict, save_path=None):
        """绘制逐时间步 RMSE 曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_name, results in results_dict.items():
            if 'per_step_rmse' in results:
                step_rmse = results['per_step_rmse']
                steps = np.arange(1, len(step_rmse) + 1)
                ax.plot(steps, step_rmse, marker='o', label=self.model_labels.get(model_name, model_name),
                       color=self.colors.get(model_name, 'gray'), linewidth=2, markersize=6)
        
        ax.set_xlabel('Forecast Hour', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title('Forecast Accuracy Degradation Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(np.arange(1, 25, 2))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved stepwise RMSE plot to {save_path}")
        plt.show()
    
    def plot_contribution_analysis(self, results_dict, save_path=None):
        """绘制 FGL 和 FiLM 的贡献分析"""
        
        if 'full' not in results_dict or not all(k in results_dict for k in ['baseline', 'no_fgl', 'no_film']):
            logger.warning("Missing some models for contribution analysis (need all 4). Skipping.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('FGL & FiLM Contribution Analysis (Overall RMSE)', fontsize=16, fontweight='bold')
        
        # 提取 RMSE 值
        baseline_rmse = results_dict['baseline'].get('overall_rmse', np.nan)
        no_fgl_rmse = results_dict['no_fgl'].get('overall_rmse', np.nan)
        no_film_rmse = results_dict['no_film'].get('overall_rmse', np.nan)
        full_rmse = results_dict['full'].get('overall_rmse', np.nan)
        
        # 1. 独立贡献 (相比于 Baseline，误差下降了多少)
        # no_fgl 模型包含 FiLM，所以其相比 Baseline 的下降即为 FiLM 的独立贡献
        film_contribution = baseline_rmse - no_fgl_rmse
        # no_film 模型包含 FGL，所以其相比 Baseline 的下降即为 FGL 的独立贡献
        fgl_contribution = baseline_rmse - no_film_rmse
        
        contribs = [film_contribution, fgl_contribution]
        labels = ['FiLM Impact', 'FGL Impact']
        colors_contrib = ['green' if v > 0 else 'red' for v in contribs]
        
        bars = ax1.bar(labels, contribs, color=colors_contrib, alpha=0.7, edgecolor='black')
        for bar, contrib in zip(bars, contribs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{contrib:.4f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_ylabel('RMSE Error Reduction', fontsize=11, fontweight='bold')
        ax1.set_title('Independent Contributions (Higher is Better)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 协同效应
        # 如果两者完全解耦，那么总预期下降: film_contribution + fgl_contribution
        # 实际总下降: baseline_rmse - full_rmse
        # 协同效应 = 实际总下降 - 预期总下降
        actual_improvement = baseline_rmse - full_rmse
        expected_improvement = film_contribution + fgl_contribution
        synergy = actual_improvement - expected_improvement
        
        bars = ax2.bar(['Synergy'], [synergy], color=['green' if synergy > 0 else 'red'], alpha=0.7, edgecolor='black', width=0.4)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{synergy:.4f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_ylabel('Synergy Value', fontsize=11, fontweight='bold')
        ax2.set_title('Synergistic Effect (Positive=Extra Improvement)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved contribution analysis plot to {save_path}")
        plt.show()
    
    def generate_all_plots(self, results_dict, output_dir):
        """生成所有对比图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.plot_rmse_comparison(results_dict, 
                                 os.path.join(output_dir, f'rmse_comparison_{timestamp}.png'))
        self.plot_mae_comparison(results_dict,
                                os.path.join(output_dir, f'mae_comparison_{timestamp}.png'))
        self.plot_stepwise_rmse(results_dict,
                               os.path.join(output_dir, f'stepwise_rmse_{timestamp}.png'))
        self.plot_contribution_analysis(results_dict,
                                       os.path.join(output_dir, f'contribution_analysis_{timestamp}.png'))
        
        logger.info(f"All plots saved to {output_dir}")


def main():
    import pickle
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ablation study results")
    parser.add_argument('--results-file', type=str, help='Path to pickled results dict')
    parser.add_argument('--output-dir', type=str, default=str(EvalConfig.results_dir),
                        help='Directory to save plots')
    args = parser.parse_args()
    
    visualizer = AblationVisualizer(config=EvalConfig)
    
    # 硬编码结果文件路径
    results_file = project_root / "ablation_study" / "results" / "ablation_results_20260519_141236.npz"
    
    if args.results_file:
        results_file = Path(args.results_file)
        
    if results_file.exists():
        # 读取 .npz 文件
        results_npz = np.load(results_file, allow_pickle=True)
        # 转换为嵌套字典以便绘图使用
        results = {}
        for key in results_npz.files:
            results[key] = results_npz[key].item()
            
        visualizer.generate_all_plots(results, args.output_dir)
    else:
        logger.warning(f"Results file not found: {results_file}")
        logger.info("Provide a valid file using --results-file")


if __name__ == "__main__":
    main()
