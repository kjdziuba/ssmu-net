#!/usr/bin/env python
"""
Entry script for SSMU-Net evaluation
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ssmu_net.eval import Evaluator
from ssmu_net.data import NpzCoreDataset
from ssmu_net.utils import load_config
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSMU-Net model")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--npz-dir", type=str, 
                       help="Directory containing NPZ files to evaluate")
    parser.add_argument("--npz-files", type=str, nargs='+',
                       help="Specific NPZ files to evaluate")
    parser.add_argument("--output", type=str, default="outputs/evaluation",
                       help="Output directory for results")
    parser.add_argument("--visualize", type=int, default=5,
                       help="Number of cores to visualize")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Determine NPZ files to evaluate
    if args.npz_dir:
        npz_dir = Path(args.npz_dir)
        npz_files = sorted(npz_dir.glob("*.npz"))
        npz_files = [str(f) for f in npz_files if not f.name.startswith('.')]
    elif args.npz_files:
        npz_files = args.npz_files
    else:
        # Default: use all NPZ files from preprocessing
        npz_dir = Path(cfg['runtime_paths']['npz'])
        npz_files = sorted(npz_dir.glob("core_*.npz"))
        npz_files = [str(f) for f in npz_files]
    
    if not npz_files:
        print("No NPZ files found to evaluate")
        return
    
    print(f"Found {len(npz_files)} cores to evaluate")
    
    # Load z-score stats if available
    stats_path = Path(cfg['runtime_paths']['tables']) / 'zscore_stats.csv'
    if stats_path.exists():
        print(f"Loading z-score stats from {stats_path}")
        stats_df = pd.read_csv(stats_path)
        z_mean = stats_df['mean'].values.astype(np.float32)
        z_std = stats_df['std'].values.astype(np.float32)
    else:
        print("No z-score stats found, proceeding without normalization")
        z_mean = None
        z_std = None
    
    # Create evaluator
    print(f"Loading model from {args.checkpoint}")
    evaluator = Evaluator(cfg, args.checkpoint)
    
    # Create dataset
    dataset = NpzCoreDataset(
        npz_files,
        mode='test',
        augment=False,
        ignore_index=-100,
        z_mean=z_mean,
        z_std=z_std
    )
    
    # Evaluate
    print("\nStarting evaluation...")
    metrics = evaluator.evaluate_dataset(dataset)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall mIoU: {metrics.get('miou_mean', 0):.4f} ± {metrics.get('miou_std', 0):.4f}")
    print(f"Overall Dice: {metrics.get('dice_mean', 0):.4f} ± {metrics.get('dice_std', 0):.4f}")
    print(f"Accuracy: {metrics.get('accuracy_mean', 0):.4f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_dir}")
    
    # Save predictions
    evaluator.save_predictions(output_dir / 'predictions')
    
    # Save metrics
    evaluator.save_metrics_report(metrics, output_dir / 'metrics.json')
    
    # Generate visualizations
    if args.visualize > 0:
        print(f"Generating visualizations for {args.visualize} cores...")
        for i in range(min(args.visualize, len(dataset))):
            evaluator.visualize_core(i, output_dir / f'core_{i}.png')
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    evaluator.generate_confusion_matrix(output_dir / 'confusion_matrix.png')
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()