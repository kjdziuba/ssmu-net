#!/usr/bin/env python
"""
Entry script for SSMU-Net model auditing
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ssmu_net.audit import run_full_audit
from ssmu_net.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Audit SSMU-Net model")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint (default: best from fold 0)")
    parser.add_argument("--n-cores", type=int, default=10,
                       help="Number of cores to use for audit (default: 10)")
    
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
    
    # Get test NPZ files
    npz_dir = Path(cfg['runtime_paths']['npz'])
    all_npz = sorted(npz_dir.glob("core_*.npz"))
    
    # Use subset for audit (it's computationally expensive)
    test_npz = [str(f) for f in all_npz[:args.n_cores]]
    
    if not test_npz:
        print("No NPZ files found. Run preprocessing first.")
        return
    
    print(f"Running audit on {len(test_npz)} cores")
    
    # Run audit
    report = run_full_audit(cfg, str(checkpoint_path), test_npz)
    
    # Print final status
    if report['audit_passed']:
        print("\n✅ MODEL PASSED AUDIT")
    else:
        print("\n❌ MODEL FAILED AUDIT")
        print("Check the audit report for details")


if __name__ == "__main__":
    main()