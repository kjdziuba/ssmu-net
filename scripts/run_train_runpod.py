#!/usr/bin/env python
"""
RunPod training entry script for SSMU-Net
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ssmu_net.train import train_cross_validation
from ssmu_net.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train SSMU-Net on RunPod")
    parser.add_argument("--config", type=str, default="ssmu_net/config_runpod.yaml",
                       help="Path to config file")
    parser.add_argument("--debug", action="store_true", 
                       help="Debug mode (fewer epochs)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Debug mode adjustments
    if args.debug:
        print("🐛 Running in debug mode")
        cfg['optim']['epochs'] = 5
        cfg['data']['folds'] = 2
    
    print(f"🚀 Starting training on RunPod")
    print(f"Config: {args.config}")
    print(f"Folds: {cfg['data']['folds']}")
    print(f"Epochs: {cfg['optim']['epochs']}")
    
    # Run training
    results = train_cross_validation(cfg)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    for fold, metrics in results.items():
        if isinstance(metrics, dict) and 'best_miou' in metrics:
            print(f"Fold {fold}: mIoU={metrics['best_miou']:.3f}, Dice={metrics['best_dice']:.3f}")
    
    print(f"\nModels saved to: {cfg['runtime_paths']['models']}")


if __name__ == "__main__":
    main()