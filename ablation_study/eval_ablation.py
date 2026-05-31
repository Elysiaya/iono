"""
消融实验评估脚本
在平静期和磁暴期分别量化各模型变体的性能，分析 FGL 和 FiLM 的独立与协同贡献
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from iono.dataset_fgl import IonosphereDatasetFGL
from data_pipeline.fgl_normalize_transform import fgl_normalize_transform
from ablation_study.ablation_models import BaselineModel, NoFGLModel, NoFiLMModel
from iono.model_fgl import StudentForecaster
from iono.config import Config
from ablation_study.ablation_config import EvalConfig

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation_eval.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AblationEvaluator:
    """消融实验评估器"""
    
    def __init__(self, config=None):
        self.config = config or EvalConfig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def load_model(self, model_name, checkpoint_path):
        """加载训练好的模型"""
        logger.info(f"Loading {model_name} from {checkpoint_path}")
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint['model_state_dict']

        # 去除 'module.' 前缀（如果是 DataParallel 保存的模型）
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
        
        # 动态推断 num_layers
        inferred_num_layers = self.config.num_layers
        layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('encoder_cells.') and k.split('.')[1].isdigit()]
        if layer_indices:
            inferred_num_layers = max(layer_indices) + 1
            
        kwargs = {
            'in_channels': self.config.in_channels,
            'hidden_channels': self.config.hidden_channels,
            'num_layers': inferred_num_layers,
            'pred_steps': self.config.pred_steps,
            'priv_gru_hidden': self.config.priv_gru_hidden,
        }
        
        if model_name == 'baseline':
            model = BaselineModel(num_aux=0, **kwargs).to(self.device)
        elif model_name == 'no_fgl':
            model = NoFGLModel(num_aux=5, **kwargs).to(self.device)
        elif model_name == 'no_film':
            model = NoFiLMModel(num_aux=5, **kwargs).to(self.device)
        elif model_name == 'full':
            model = StudentForecaster(num_aux=5, **kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # 多 GPU 支持
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        logger.info(f"Model {model_name} loaded successfully")
        return model
    
    def build_val_loader(self):
        """构建验证数据加载器"""
        dataset = IonosphereDatasetFGL(
            self.config.hickle_paths,
            window_size=self.config.window_size,
            future_size=self.config.future_size,
            pred_steps=self.config.pred_steps,
            transform=fgl_normalize_transform,
            return_time=True
        )
        
        total = len(dataset)
        train_size = int(0.9 * total)
        val_dataset = Subset(dataset, range(train_size, total))
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        
        logger.info(f"Validation set size: {len(val_dataset)}")
        return val_loader, val_dataset
    
    def evaluate_model(self, model, val_loader, model_name, use_film=True):
        """评估单个模型"""
        logger.info(f"Evaluating {model_name}...")
        
        all_preds = []
        all_targets = []
        all_aux_future = []  # 用于后续筛选极端气象
        all_target_times = []
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            val_pbar = tqdm(val_loader, desc=f"Evaluating {model_name}", leave=False)
            for X_hist, aux_hist, X_future, aux_future, y, target_time in val_pbar:
                X_hist, aux_hist = X_hist.to(self.device), aux_hist.to(self.device)
                X_future, aux_future = X_future.to(self.device), aux_future.to(self.device)
                y = y.to(self.device)
                
                # Baseline 没用 future_aux, FiLM 和 Full 才用
                pred = model(
                    X_hist,
                    aux_x=aux_hist,
                    future_aux=aux_future if use_film else None,
                    dec_aux=aux_future[:, :self.config.pred_steps, :],
                    tf_ratio=0.0
                )
                
                # 处理返回值（可能是 tuple）
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_aux_future.append(aux_future.cpu().numpy())
                all_target_times.extend(target_time)
        
        preds = np.concatenate(all_preds, axis=0)  # (N, T, C, H, W) 或 (N, T, 1, H, W)
        targets = np.concatenate(all_targets, axis=0)
        aux_futures = np.concatenate(all_aux_future, axis=0)
        
        logger.info(f"Prediction shape: {preds.shape}, Target shape: {targets.shape}")
        
        # 计算不同情况下的 RMSE
        model_results = self.calculate_daily_metrics(preds, targets, aux_futures, all_target_times)
        self.results[model_name] = model_results
        
        return model_results
        
    def calculate_daily_metrics(self, preds, targets, aux_futures, target_times):
        """按天计算RMSE, CC 和最低Dst"""
        import pandas as pd
        from collections import defaultdict
        
        pred_steps = preds.shape[1]
        dst_values = aux_futures[:, :pred_steps, 1] * 250.0 - 200.0  # (N, T)
        
        daily_stats = defaultdict(list)
        
        # 将 target_times 解析为年月日，归类 index
        for i, t_str in enumerate(target_times):
            dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
            day_key = dt.strftime("%Y-%m-%d")
            daily_stats[day_key].append(i)
        
        metrics = {}
        daily_results_list = []
        
        for day_key in sorted(daily_stats.keys()):
            indices = daily_stats[day_key]
            
            day_preds = preds[indices] * 100.0
            day_targets = targets[indices] * 100.0
            day_dst = dst_values[indices]  # (N, T)
            
            # 计算各项指标
            rmse = np.sqrt(np.mean((day_preds - day_targets)**2))
            cc = np.corrcoef(day_preds.flatten(), day_targets.flatten())[0, 1]
            min_dst = np.min(day_dst)
            
            metrics[f'{day_key}_rmse'] = rmse
            metrics[f'{day_key}_cc'] = cc
            metrics[f'{day_key}_dst'] = min_dst
            
            daily_results_list.append({
                'Date': day_key,
                'RMSE': rmse,
                'CC': cc,
                'Min_Dst': min_dst
            })
            
        metrics['_daily_results_list'] = daily_results_list
        return metrics
        
    def run_evaluation(self, checkpoint_dict):
        """主评估流程
        
        Args:
            checkpoint_dict: 字典 {model_name: checkpoint_path}
        """
        val_loader, _ = self.build_val_loader()
        
        for model_name, ckpt_path in checkpoint_dict.items():
            if not os.path.exists(ckpt_path):
                logger.warning(f"Checkpoint not found for {model_name}: {ckpt_path}")
                continue
                
            use_film = model_name in ['no_fgl', 'full']
            model = self.load_model(model_name, ckpt_path)
            self.evaluate_model(model, val_loader, model_name, use_film=use_film)
            
            # 清理显存
            del model
            torch.cuda.empty_cache()
            
        self.save_results()
        self.print_summary()
        
    def save_results(self):
        """保存评估结果为 NPZ 和 CSV"""
        import pandas as pd
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 提取用于保存NPZ的数据（过滤掉不能存入NPZ的_daily_results_list）
        npz_results = {}
        for model_name, res in self.results.items():
            npz_results[model_name] = {k: v for k, v in res.items() if k != '_daily_results_list'}
            
        save_path = self.config.results_dir / f"ablation_results_{timestamp}.npz"
        np.savez(save_path, **npz_results)
        logger.info(f"Results saved to {save_path}")
        
        # 保存为 CSV
        csv_rows = []
        for model_name, res in self.results.items():
            daily_list = res.get('_daily_results_list', [])
            for d_data in daily_list:
                row = {'Model': model_name}
                row.update(d_data)
                csv_rows.append(row)
        
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_path = self.config.results_dir / f"ablation_results_daily_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Daily results saved to {csv_path}")
        
    def print_summary(self):
        """打印总结表格"""
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY RESULTS SUMMARY (DAILY BASED)")
        logger.info("="*80)
        
        if not self.results:
            return

        print(f"{'Model':<15} | {'Quiet(RMSE)':<12} | {'Quiet(CC)':<12} | {'Storm(RMSE)':<12} | {'Storm(CC)':<12}")
        print("-" * 80)
        
        for model_name, res in self.results.items():
            daily_list = res.get('_daily_results_list', [])
            
            quiet_rmses, quiet_ccs = [], []
            storm_rmses, storm_ccs = [], []
            
            for d in daily_list:
                if d['Min_Dst'] > -30:
                    quiet_rmses.append(d['RMSE'])
                    quiet_ccs.append(d['CC'])
                elif d['Min_Dst'] <= -50:  # <-50 nT 定义为磁暴日
                    storm_rmses.append(d['RMSE'])
                    storm_ccs.append(d['CC'])
            
            q_rmse = np.mean(quiet_rmses) if quiet_rmses else np.nan
            q_cc = np.mean(quiet_ccs) if quiet_ccs else np.nan
            s_rmse = np.mean(storm_rmses) if storm_rmses else np.nan
            s_cc = np.mean(storm_ccs) if storm_ccs else np.nan
            
            print(f"{model_name:<15} | {q_rmse:<12.4f} | {q_cc:<12.4f} | {s_rmse:<12.4f} | {s_cc:<12.4f}")
            
        logger.info("\n* Note: Quiet day defined as Min Dst > -30 nT; Storm day defined as Min Dst <= -50 nT")


if __name__ == "__main__":
    import argparse

    def latest_best_checkpoint(root, pattern):
        candidates = sorted(Path(root).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="", help="Path to baseline best checkpoint")
    parser.add_argument("--no_fgl", type=str, default="", help="Path to no_fgl best checkpoint")
    parser.add_argument("--no_film", type=str, default="", help="Path to no_film best checkpoint")
    parser.add_argument("--full", type=str, default="", help="Path to full (StudentForecaster) best checkpoint")
    args = parser.parse_args()

    legacy_ablation_dir = project_root / "ablation_study" / "ablation_checkpoints"
    
    ckpt_dict = {
        'baseline': args.baseline
            or latest_best_checkpoint(EvalConfig.ablation_dir, "baseline_*/best_baseline.pth")
            or latest_best_checkpoint(legacy_ablation_dir, "baseline_*/best_baseline.pth"),
        'no_fgl': args.no_fgl
            or latest_best_checkpoint(EvalConfig.ablation_dir, "no_fgl_*/best_no_fgl.pth")
            or latest_best_checkpoint(legacy_ablation_dir, "no_fgl_*/best_no_fgl.pth"),
        'no_film': args.no_film
            or latest_best_checkpoint(EvalConfig.ablation_dir, "no_film_*/best_no_film.pth")
            or latest_best_checkpoint(legacy_ablation_dir, "no_film_*/best_no_film.pth"),
        'full': args.full or Config.student_checkpoint,
    }

    # 移除仍为空的项或者不存在的文件
    ckpt_dict = {k: v for k, v in ckpt_dict.items() if v and Path(v).exists()}
    
    if not ckpt_dict:
        logger.error("No checkpoints provided or Auto-find failed for evaluation. Use --baseline, --no_fgl, etc to provide paths.")
        sys.exit(1)
        
    evaluator = AblationEvaluator()
    evaluator.run_evaluation(ckpt_dict)
