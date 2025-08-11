#!/usr/bin/env python
"""
Script to generate publication-ready figures for SSMU-Net results
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssmu_net.figures import generate_paper_figures
from ssmu_net.utils import load_config


# Define all available figures
ALL_FIGS = ['arch', 'bands', 'saliency', 'qualitative', 'ablation', 
            'band_vs_miou', 'throughput', 'occlusion']


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for SSMU-Net"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ssmu_net/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--dfir-dir",
        type=str,
        help="Directory containing DFIR results (optional)"
    )
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        choices=['arch', 'bands', 'saliency', 'qualitative', 'ablation', 
                 'band_vs_miou', 'throughput', 'occlusion', 'all'],
        default=['all'],
        help="Specific figures to generate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)
    
    # Override figure list based on command line arguments
    if 'all' in args.figures:
        # Force generation of ALL figures
        cfg.setdefault('reporting', {})['figures'] = ALL_FIGS
        print(f"Generating ALL figures: {', '.join(ALL_FIGS)}")
    else:
        # Generate only specific requested figures
        cfg.setdefault('reporting', {})['figures'] = args.figures
        print(f"Generating specific figures: {', '.join(args.figures)}")
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Generate figures
    try:
        figures = generate_paper_figures(cfg, str(output_dir), args.dfir_dir)
        
        print("\n✓ Figure generation complete!")
        print(f"  Generated {len(figures)} figures:")
        for fig_type, path in figures.items():
            print(f"    - {fig_type}: {path}")
            
    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()