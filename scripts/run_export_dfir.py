#!/usr/bin/env python
"""
Entry script for DFIR band export
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ssmu_net.export import export_dfir_bands
from ssmu_net.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Export DFIR bands from SSMU-Net")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint (default: best from fold 0)")
    parser.add_argument("--output", type=str, default="outputs/dfir",
                       help="Output directory for DFIR specifications")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Default to best model from fold 0
        checkpoint_path = Path(cfg['runtime_paths']['models']) / 'fold_0' / 'checkpoint_best.pth'
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            print("Please train a model first or specify --checkpoint")
            return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Export DFIR bands
    summary = export_dfir_bands(cfg, str(checkpoint_path), args.output)
    
    print("\n✅ DFIR export complete!")
    print(f"Specifications saved to: {args.output}")


if __name__ == "__main__":
    main()