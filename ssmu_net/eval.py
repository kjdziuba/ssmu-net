"""
Evaluation script for SSMU-Net models
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from .models import create_model
from .data import NpzCoreDataset
from .evaluation_metrics import compute_metrics, aggregate_metrics
from .utils import load_config


class Evaluator:
    """Evaluator for SSMU-Net models on full cores"""
    
    def __init__(self, cfg: Dict, checkpoint_path: str = None):
        """
        Args:
            cfg: Configuration dictionary
            checkpoint_path: Optional path to model checkpoint
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model(cfg).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Class names and colors for visualization
        self.class_names = [
            'Background', 'Normal Epithelium', 'Normal Stroma', 
            'Cancer Epithelium', 'Cancer Stroma', 'Blood', 
            'Necrosis', 'Immune'
        ]
        
        self.class_colors = [
            (0, 0, 0),            # Background
            (0, 255, 0),          # Normal Epithelium
            (128, 0, 128),        # Normal Stroma
            (255, 0, 255),        # Cancer Epithelium
            (0, 0, 255),          # Cancer-Associated Stroma
            (255, 0, 0),          # Blood
            (255, 165, 0),        # Necrosis
            (255, 255, 0),        # Immune
        ]
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's just the state dict
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Return checkpoint info if available
        if isinstance(checkpoint, dict):
            return checkpoint.get('metrics', {}), checkpoint.get('epoch', -1)
        return {}, -1
    
    @torch.no_grad()
    def predict_core(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict segmentation for a full core
        
        Args:
            X: Input tensor (C, H, W)
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        # Add batch dimension
        X = X.unsqueeze(0).to(self.device)
        
        # Forward pass
        if self.cfg['optim'].get('amp', False):
            with autocast():
                logits, _ = self.model(X)
        else:
            logits, _ = self.model(X)
        
        # Get probabilities and predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Remove batch dimension
        preds = preds.squeeze(0)
        probs = probs.squeeze(0)
        
        return preds, probs
    
    def evaluate_dataset(self, dataset: NpzCoreDataset) -> Dict[str, float]:
        """
        Evaluate model on full dataset
        
        Args:
            dataset: NPZ dataset with full cores
        
        Returns:
            Dictionary of aggregated metrics
        """
        all_metrics = []
        core_results = []
        
        pbar = tqdm(range(len(dataset)), desc="Evaluating cores")
        
        for idx in pbar:
            # Get core data
            batch = dataset[idx]
            X = batch['X']
            y = batch['y']
            core_id = batch['core_id']
            
            # Predict
            preds, probs = self.predict_core(X)
            
            # Move to CPU for metrics
            preds = preds.cpu()
            y = y.cpu()
            
            # Compute metrics for this core
            metrics = compute_metrics(
                preds, y, 
                num_classes=self.cfg['model']['classes'],
                exclude_background=True
            )
            
            # Add core info
            metrics['core_id'] = core_id
            all_metrics.append(metrics)
            
            # Store for detailed analysis
            core_results.append({
                'core_id': core_id,
                'predictions': preds.numpy(),
                'ground_truth': y.numpy(),
                'metrics': metrics
            })
            
            # Update progress bar
            pbar.set_postfix({'mIoU': f"{metrics['miou']:.3f}"})
        
        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics)
        
        # Add per-core statistics
        mious = [m['miou'] for m in all_metrics]
        aggregated['miou_per_core_mean'] = float(np.mean(mious))
        aggregated['miou_per_core_std'] = float(np.std(mious))
        aggregated['miou_per_core_min'] = float(np.min(mious))
        aggregated['miou_per_core_max'] = float(np.max(mious))
        
        # Store detailed results
        self.core_results = core_results
        
        return aggregated
    
    def save_predictions(self, output_dir: str):
        """Save predictions for all evaluated cores"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in self.core_results:
            core_id = result['core_id']
            preds = result['predictions']
            
            # Save as NPZ
            np.savez_compressed(
                output_dir / f'{core_id}_predictions.npz',
                predictions=preds,
                ground_truth=result['ground_truth'],
                metrics=result['metrics']
            )
    
    def visualize_core(self, core_idx: int, save_path: Optional[str] = None):
        """
        Visualize predictions for a specific core
        
        Args:
            core_idx: Index of core in results
            save_path: Optional path to save figure
        """
        if not hasattr(self, 'core_results'):
            raise ValueError("No evaluation results. Run evaluate_dataset first.")
        
        result = self.core_results[core_idx]
        preds = result['predictions']
        gt = result['ground_truth']
        metrics = result['metrics']
        
        # Create colormap
        colors = np.array(self.class_colors) / 255.0
        cmap = ListedColormap(colors)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        ax = axes[0]
        im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=7)
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        # Predictions
        ax = axes[1]
        ax.imshow(preds, cmap=cmap, vmin=0, vmax=7)
        ax.set_title(f'Predictions (mIoU={metrics["miou"]:.3f})')
        ax.axis('off')
        
        # Difference map
        ax = axes[2]
        diff = (preds != gt).astype(float)
        # Mask out ignored pixels
        diff[gt == -100] = np.nan
        ax.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title('Error Map')
        ax.axis('off')
        
        # Add colorbar for classes
        cbar = fig.colorbar(im, ax=axes[:2], orientation='horizontal', 
                           fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(8))
        cbar.set_ticklabels(self.class_names, rotation=45, ha='right')
        
        plt.suptitle(f'Core: {result["core_id"]}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_confusion_matrix(self, save_path: Optional[str] = None):
        """Generate and save confusion matrix from all predictions"""
        if not hasattr(self, 'core_results'):
            raise ValueError("No evaluation results. Run evaluate_dataset first.")
        
        # Collect all predictions and ground truths
        all_preds = []
        all_gt = []
        
        for result in self.core_results:
            preds = result['predictions'].flatten()
            gt = result['ground_truth'].flatten()
            
            # Filter out ignored pixels
            valid = gt != -100
            all_preds.append(preds[valid])
            all_gt.append(gt[valid])
        
        all_preds = np.concatenate(all_preds)
        all_gt = np.concatenate(all_gt)
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_gt, all_preds, labels=list(range(8)))
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Normalized Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return cm, cm_norm
    
    def save_metrics_report(self, metrics: Dict, output_path: str):
        """Save detailed metrics report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report
        report = {
            'overall_metrics': {
                k: v for k, v in metrics.items() 
                if not k.endswith('_mean') and not k.endswith('_std')
            },
            'per_class_iou': {},
            'per_class_dice': {},
            'statistics': {}
        }
        
        # Extract per-class metrics
        for key, value in metrics.items():
            if key.startswith('iou_class_'):
                class_idx = int(key.split('_')[-1].replace('_mean', ''))
                report['per_class_iou'][self.class_names[class_idx]] = value
            elif key.startswith('dice_class_'):
                class_idx = int(key.split('_')[-1].replace('_mean', ''))
                report['per_class_dice'][self.class_names[class_idx]] = value
        
        # Add statistics
        if hasattr(self, 'core_results'):
            mious = [r['metrics']['miou'] for r in self.core_results]
            report['statistics'] = {
                'num_cores': len(self.core_results),
                'miou_mean': float(np.mean(mious)),
                'miou_std': float(np.std(mious)),
                'miou_min': float(np.min(mious)),
                'miou_max': float(np.max(mious)),
                'miou_median': float(np.median(mious))
            }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as CSV for easy viewing
        csv_path = output_path.with_suffix('.csv')
        if hasattr(self, 'core_results'):
            df_data = []
            for r in self.core_results:
                row = {'core_id': r['core_id']}
                row.update(r['metrics'])
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
        
        print(f"Saved metrics report to {output_path}")


def evaluate_fold(cfg: Dict, fold: int, test_paths: List[str]) -> Dict[str, float]:
    """
    Evaluate a single fold's best model
    
    Args:
        cfg: Configuration
        fold: Fold index
        test_paths: List of test NPZ paths
    
    Returns:
        Dictionary of metrics
    """
    # Setup paths
    model_dir = Path(cfg['runtime_paths']['models']) / f'fold_{fold}'
    checkpoint_path = model_dir / 'checkpoint_best.pth'
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found for fold {fold}")
        return {}
    
    # Create evaluator
    evaluator = Evaluator(cfg, str(checkpoint_path))
    
    # Load z-score stats from training
    stats_path = Path(cfg['runtime_paths']['tables']) / 'zscore_stats.csv'
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        z_mean = stats_df['mean'].values.astype(np.float32)
        z_std = stats_df['std'].values.astype(np.float32)
    else:
        z_mean = None
        z_std = None
    
    # Create test dataset
    test_dataset = NpzCoreDataset(
        test_paths,
        mode='test',
        augment=False,
        ignore_index=0,
        z_mean=z_mean,
        z_std=z_std
    )
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(test_dataset)
    
    # Save results
    output_dir = Path(cfg['runtime_paths']['tables']) / f'fold_{fold}_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    evaluator.save_predictions(output_dir / 'predictions')
    
    # Save metrics
    evaluator.save_metrics_report(metrics, output_dir / 'metrics.json')
    
    # Generate visualizations
    fig_dir = Path(cfg['runtime_paths']['figures']) / f'fold_{fold}_eval'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize a few cores
    for i in range(min(5, len(test_dataset))):
        evaluator.visualize_core(i, fig_dir / f'core_{i}.png')
    
    # Generate confusion matrix
    evaluator.generate_confusion_matrix(fig_dir / 'confusion_matrix.png')
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SSMU-Net model")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--npz", type=str, nargs='+', required=True,
                       help="NPZ files to evaluate")
    parser.add_argument("--output", type=str, default="outputs/evaluation",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Create evaluator
    evaluator = Evaluator(cfg, args.checkpoint)
    
    # Create dataset
    dataset = NpzCoreDataset(
        args.npz,
        mode='test',
        augment=False,
        ignore_index=-100
    )
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(dataset)
    
    # Save results
    output_dir = Path(args.output)
    evaluator.save_predictions(output_dir / 'predictions')
    evaluator.save_metrics_report(metrics, output_dir / 'metrics.json')
    
    # Visualizations
    for i in range(min(3, len(dataset))):
        evaluator.visualize_core(i, output_dir / f'core_{i}.png')
    
    evaluator.generate_confusion_matrix(output_dir / 'confusion_matrix.png')
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")