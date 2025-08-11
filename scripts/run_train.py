#!/usr/bin/env python
"""
Entry script for SSMU-Net training
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ssmu_net.train import train_cross_validation
from ssmu_net.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train SSMU-Net with cross-validation")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml",
                       help="Path to config file")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode with reduced epochs")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Debug mode: reduce epochs for quick testing
    if args.debug:
        print("Running in DEBUG mode - reduced epochs")
        cfg['optim']['epochs'] = 2
        cfg['data']['folds'] = 2
        cfg['optim']['patience'] = 1
    
    # Run training
    train_cross_validation(cfg)


if __name__ == "__main__":
    main()