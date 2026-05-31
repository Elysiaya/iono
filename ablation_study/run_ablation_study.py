"""
消融实验主控脚本
协调完整的消融实验流程：训练 → 评估 → 可视化
"""

import os
import sys
from pathlib import Path
import pickle
import argparse
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ablation_study.ablation_config import EvalConfig
from ablation_study.eval_ablation import AblationEvaluator
from ablation_study.visualize_ablation import AblationVisualizer

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation_study.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_complete_ablation_study(checkpoint_dir=None, output_dir=None, skip_train=False, skip_eval=False):
    """
    运行完整的消融实验流程
    
    Args:
        checkpoint_dir: 检查点目录（如果 skip_train=True，用于加载模型）
        output_dir: 结果输出目录
        skip_train: 是否跳过训练阶段
        skip_eval: 是否跳过评估阶段
    """
    
    checkpoint_dir = checkpoint_dir or str(EvalConfig.ablation_dir)
    output_dir = output_dir or str(EvalConfig.results_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Starting Complete Ablation Study Pipeline")
    logger.info("="*80)
    
    # ==================== 阶段 1: 训练 ====================
    if not skip_train:
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: Training Ablation Models")
        logger.info("="*80)
        
        try:
            import subprocess
            
            for model_name in ['baseline', 'no_fgl', 'no_film', 'full']:
                logger.info(f"\nTraining {model_name}...")
                
                # Each model has its own distinct training script now
                script_path = project_root / "ablation_study" / f"train_{model_name}.py"
                
                if script_path.exists():
                    subprocess.run([sys.executable, str(script_path)], check=True)
                    logger.info(f"✓ Training {model_name} completed")
                else:
                    logger.warning(f"Training script {script_path} not found for {model_name}.")
        
        except Exception as e:
            logger.error(f"Error during training phase: {e}")
            logger.error("Skipping to evaluation phase...")
    
    # ==================== 阶段 2: 评估 ====================
    if not skip_eval:
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: Evaluating Models")
        logger.info("="*80)
        
        try:
            evaluator = AblationEvaluator(config=EvalConfig)
            results = evaluator.evaluate_ablation_study(checkpoint_dir)
            
            # 保存结果为 pickle，便于后续可视化
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_pkl_path = os.path.join(output_dir, f"ablation_results_{timestamp}.pkl")
            
            with open(results_pkl_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"✓ Results saved to {results_pkl_path}")
            
            # 保存文本报告和 CSV
            evaluator.save_results(output_dir)
            
        except Exception as e:
            logger.error(f"Error during evaluation phase: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ==================== 阶段 3: 可视化 ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: Generating Visualizations")
    logger.info("="*80)
    
    try:
        # 查找最新的结果文件
        results_pkl_files = sorted(Path(output_dir).glob("ablation_results_*.pkl"), 
                                  key=lambda x: x.stat().st_mtime, reverse=True)
        
        if results_pkl_files:
            latest_results_file = str(results_pkl_files[0])
            logger.info(f"Loading results from {latest_results_file}")
            
            with open(latest_results_file, 'rb') as f:
                results = pickle.load(f)
            
            visualizer = AblationVisualizer(config=EvalConfig)
            visualizer.generate_all_plots(results, output_dir)
            
            logger.info("✓ Visualizations generated successfully")
        else:
            logger.warning("No results pickle file found. Skipping visualization.")
    
    except Exception as e:
        logger.error(f"Error during visualization phase: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("Complete Ablation Study Pipeline Finished!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ablation study pipeline"
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory for model checkpoints')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory for results')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training phase')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation phase')
    parser.add_argument('--phase', type=str, 
                       choices=['train', 'eval', 'viz', 'all'],
                       default='all',
                       help='Which phase to run')
    
    args = parser.parse_args()
    
    # 根据 --phase 参数调整跳过行为
    skip_train = args.skip_train or args.phase in ['eval', 'viz']
    skip_eval = args.phase == 'viz'  # 可视化不需要重新评估，只需加载结果
    
    run_complete_ablation_study(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        skip_train=skip_train,
        skip_eval=skip_eval
    )


if __name__ == "__main__":
    main()
