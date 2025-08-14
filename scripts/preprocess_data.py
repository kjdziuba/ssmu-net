#!/usr/bin/env python3
"""
Optimized preprocessing with statistics and visualization
- Skips cores without annotations
- Generates preprocessing statistics
- Creates before/after spectral plots
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssmu_net.preprocess_optimized import process_core, save_label_map
from specml.data.spectroscopic_data import SpectroscopicData


def check_annotation_exists(png_path: str) -> Tuple[bool, Dict]:
    """Check if annotation exists and has non-background pixels"""
    from PIL import Image
    
    if not os.path.exists(png_path):
        return False, {'status': 'missing'}
    
    # Load and check annotation
    img = Image.open(png_path).convert('RGBA')
    arr = np.array(img)
    
    # Check if there are any non-black pixels (potential annotations)
    non_black = np.any(arr[:, :, :3] != 0, axis=2)
    
    if not non_black.any():
        return False, {'status': 'empty', 'reason': 'all black pixels'}
    
    # Count unique colors (excluding alpha channel)
    rgb = arr[:, :, :3]
    unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)
    
    # If only black and white/gray, likely no real annotations
    if len(unique_colors) <= 2:
        colors_list = unique_colors.tolist()
        if all(c[0] == c[1] == c[2] for c in colors_list):  # All grayscale
            return False, {'status': 'grayscale_only', 'colors': len(unique_colors)}
    
    return True, {'status': 'valid', 'unique_colors': len(unique_colors)}


def plot_spectra_comparison(before_data: np.ndarray, after_data: np.ndarray, 
                           wn_before: np.ndarray, wn_after: np.ndarray,
                           core_name: str, output_dir: Path):
    """Plot before/after preprocessing comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Preprocessing Effects: {core_name}', fontsize=14, fontweight='bold')
    
    # Sample some spectra for visualization
    n_samples = min(100, before_data.shape[0] * before_data.shape[1])
    
    # Reshape to 2D (pixels x bands)
    before_flat = before_data.reshape(-1, before_data.shape[-1])
    after_flat = after_data.reshape(-1, after_data.shape[-1])
    
    # Random sampling
    idx = np.random.choice(before_flat.shape[0], min(n_samples, before_flat.shape[0]), replace=False)
    
    # 1. Raw spectra
    ax = axes[0, 0]
    for i in idx[:20]:  # Plot 20 spectra
        if np.any(before_flat[i] != 0):  # Skip zero spectra
            ax.plot(wn_before, before_flat[i], alpha=0.3, color='blue')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('Raw Spectra (20 samples)')
    ax.grid(True, alpha=0.3)
    
    # 2. After preprocessing
    ax = axes[0, 1]
    for i in idx[:20]:
        if np.any(after_flat[i] != 0):
            ax.plot(wn_after, after_flat[i], alpha=0.3, color='green')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity (2nd derivative, L2 norm)')
    ax.set_title('After Preprocessing (20 samples)')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean spectrum before/after
    ax = axes[1, 0]
    
    # Calculate mean of non-zero spectra
    nonzero_mask_before = np.any(before_flat != 0, axis=1)
    if nonzero_mask_before.any():
        mean_before = before_flat[nonzero_mask_before].mean(axis=0)
        std_before = before_flat[nonzero_mask_before].std(axis=0)
        
        ax.plot(wn_before, mean_before, 'b-', label='Mean', linewidth=2)
        ax.fill_between(wn_before, mean_before - std_before, mean_before + std_before, 
                        alpha=0.3, color='blue', label='±1 STD')
    
    # Mark wax gap region
    ax.axvspan(1350, 1490, alpha=0.2, color='red', label='Wax gap (removed)')
    ax.axvspan(1600, 1700, alpha=0.2, color='green', label='Amide I (tissue)')
    
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('Mean Spectrum (Raw)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Spectral range comparison
    ax = axes[1, 1]
    
    # Bar plot showing band reduction
    bands_data = {
        'Before': len(wn_before),
        'After': len(wn_after),
        'Removed': len(wn_before) - len(wn_after)
    }
    
    bars = ax.bar(bands_data.keys(), bands_data.values(), color=['blue', 'green', 'red'])
    ax.set_ylabel('Number of Bands')
    ax.set_title('Spectral Bands')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Add percentage for removed
    removed_pct = (bands_data['Removed'] / bands_data['Before']) * 100
    ax.text(2, bands_data['Removed']/2, f'{removed_pct:.1f}%', 
            ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'preprocessing_plots' / f'{core_name}_preprocessing.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_statistics(stats: List[Dict], output_dir: Path):
    """Generate comprehensive summary statistics and plots"""
    
    # Create summary DataFrame
    df = pd.DataFrame(stats)
    
    # Calculate aggregated statistics
    summary = {
        'total_cores': len(df),
        'processed_cores': len(df[df['status'] == 'processed']),
        'skipped_no_annotation': len(df[df['status'] == 'skipped_no_annotation']),
        'skipped_empty_annotation': len(df[df['status'] == 'skipped_empty_annotation']),
        'failed_processing': len(df[df['status'] == 'failed']),
    }
    
    # For processed cores
    processed_df = df[df['status'] == 'processed']
    if len(processed_df) > 0:
        summary.update({
            'avg_tissue_percent': processed_df['tissue_percent'].mean(),
            'std_tissue_percent': processed_df['tissue_percent'].std(),
            'min_tissue_percent': processed_df['tissue_percent'].min(),
            'max_tissue_percent': processed_df['tissue_percent'].max(),
            'total_pixels': processed_df['n_pixels'].sum(),
            'total_tissue_pixels': processed_df['n_tissue_pixels'].sum(),
            'avg_classes': processed_df['n_classes'].mean(),
        })
    
    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Summary Statistics', fontsize=16, fontweight='bold')
    
    # 1. Core processing status
    ax = axes[0, 0]
    status_counts = df['status'].value_counts()
    colors = {'processed': 'green', 'skipped_no_annotation': 'orange', 
              'skipped_empty_annotation': 'yellow', 'failed': 'red'}
    ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
           colors=[colors.get(s, 'gray') for s in status_counts.index])
    ax.set_title(f'Core Processing Status (n={len(df)})')
    
    # 2. Tissue percentage distribution
    if len(processed_df) > 0:
        ax = axes[0, 1]
        ax.hist(processed_df['tissue_percent'], bins=20, color='blue', edgecolor='black')
        ax.axvline(processed_df['tissue_percent'].mean(), color='red', 
                  linestyle='--', label=f'Mean: {processed_df["tissue_percent"].mean():.1f}%')
        ax.set_xlabel('Tissue Percentage (%)')
        ax.set_ylabel('Number of Cores')
        ax.set_title('Tissue Retention Distribution')
        ax.legend()
        
        # 3. Classes per core
        ax = axes[0, 2]
        class_counts = processed_df['n_classes'].value_counts().sort_index()
        ax.bar(class_counts.index, class_counts.values, color='green')
        ax.set_xlabel('Number of Classes')
        ax.set_ylabel('Number of Cores')
        ax.set_title('Annotation Classes per Core')
        
        # 4. Tissue pixels vs total pixels
        ax = axes[1, 0]
        ax.scatter(processed_df['n_pixels'], processed_df['n_tissue_pixels'], 
                  alpha=0.6, color='blue')
        ax.set_xlabel('Total Pixels')
        ax.set_ylabel('Tissue Pixels')
        ax.set_title('Tissue vs Total Pixels')
        
        # Add diagonal line for reference
        max_val = max(processed_df['n_pixels'].max(), processed_df['n_tissue_pixels'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='100% tissue')
        ax.legend()
        
        # 5. Core sizes
        ax = axes[1, 1]
        sizes = processed_df['shape'].apply(lambda x: eval(x)[0] * eval(x)[1])
        ax.hist(sizes, bins=20, color='purple', edgecolor='black')
        ax.set_xlabel('Core Size (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Core Size Distribution')
        
        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""Summary Statistics:
        
Cores Processed: {summary['processed_cores']}/{summary['total_cores']}
Skipped (no annotation): {summary['skipped_no_annotation']}
Skipped (empty annotation): {summary['skipped_empty_annotation']}
Failed: {summary['failed_processing']}

Tissue Retention:
  Mean: {summary.get('avg_tissue_percent', 0):.1f}%
  Std: {summary.get('std_tissue_percent', 0):.1f}%
  Range: {summary.get('min_tissue_percent', 0):.1f}% - {summary.get('max_tissue_percent', 0):.1f}%

Data Volume:
  Total pixels: {summary.get('total_pixels', 0):,}
  Tissue pixels: {summary.get('total_tissue_pixels', 0):,}
  Avg classes/core: {summary.get('avg_classes', 0):.1f}

Preprocessing:
  Threshold: 2.5 (Amide I)
  Bands: 425 → 354 (16.7% reduction)
  Derivative: 2nd order
  Normalization: Double L2"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
               fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    plot_path = output_dir / 'preprocessing_summary.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save statistics to JSON
    json_path = output_dir / 'preprocessing_statistics.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed DataFrame to CSV
    csv_path = output_dir / 'preprocessing_details.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n📊 Statistics saved to:")
    print(f"  - Summary plot: {plot_path}")
    print(f"  - JSON stats: {json_path}")
    print(f"  - Detailed CSV: {csv_path}")
    
    return summary


def main():
    # Load configuration
    config_path = 'ssmu_net/config_optimized.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup directories
    npz_dir = Path(cfg['runtime_paths']['npz'])
    npz_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(cfg['runtime_paths']['logs'])
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("OPTIMIZED PREPROCESSING WITH STATISTICS")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Tissue threshold: {cfg['preprocess']['qc_mask']['threshold']}")
    print(f"  - SG: window={cfg['preprocess']['sg']['window']}, "
          f"order={cfg['preprocess']['sg']['polyorder']}, "
          f"deriv={cfg['preprocess']['sg']['deriv']}")
    print(f"  - Double L2 normalization: {cfg['preprocess']['double_l2']}")
    print(f"  - Output: {npz_dir}")
    print("=" * 70)
    
    # Save label map
    save_label_map(str(npz_dir))
    
    # Load metadata
    metadata_df = pd.read_excel(cfg['data']['metadata_excel'])
    print(f"Loaded metadata with {len(metadata_df)} cores\n")
    
    # Statistics tracking
    stats = []
    processed = 0
    skipped_no_anno = 0
    skipped_empty_anno = 0
    failed = 0
    
    # Process each core
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing cores"):
        # Get core position
        position = row['Position'].strip()
        
        # Filter cores (C2-M16)
        letter, num = position[0], int(position[1:])
        if letter < 'C' or letter > 'M' or num < 2 or num > 16:
            continue
        
        # Get QCL grid number
        grid = int(row['qcl_grid'])
        if grid < 0:
            continue
        
        # Construct paths
        qcl_folder = row['qcl_folder']
        core_path = os.path.join(
            cfg['data']['raw_root'],
            'Isolated Cores',
            qcl_folder,
            'zarr_data',
            f'core {grid}'
        )
        
        # Annotation path
        png_path = os.path.join(
            cfg['data']['annotations_dir'],
            f'{position} anno.png'
        )
        
        # Check if annotation exists and is valid
        has_annotation, anno_info = check_annotation_exists(png_path)
        
        if not has_annotation:
            if anno_info['status'] == 'missing':
                skipped_no_anno += 1
                stats.append({
                    'core': position,
                    'status': 'skipped_no_annotation',
                    'reason': 'annotation file missing'
                })
            else:
                skipped_empty_anno += 1
                stats.append({
                    'core': position,
                    'status': 'skipped_empty_annotation',
                    'reason': anno_info.get('reason', 'no valid annotations')
                })
            continue
        
        # Output path
        output_path = npz_dir / f'core_{position}.npz'
        
        # Check if core directory exists
        if not os.path.exists(core_path):
            failed += 1
            stats.append({
                'core': position,
                'status': 'failed',
                'reason': 'core directory not found'
            })
            continue
        
        # Load raw data for before/after comparison (only for first 5 cores)
        plot_comparison = processed < 5
        
        if plot_comparison:
            try:
                # Load raw data
                sd = SpectroscopicData(file_path=core_path)
                H, W = sd.ypixels, sd.xpixels
                before_data = sd.data.reshape(H, W, -1)
                wn_before = sd.wavenumbers
            except:
                plot_comparison = False
        
        # Process core
        meta_dict = row.to_dict()
        success = process_core(
            core_path=core_path,
            png_path=png_path,
            output_path=str(output_path),
            cfg=cfg,
            logs_dir=str(logs_dir),
            meta_dict=meta_dict
        )
        
        if success:
            processed += 1
            
            # Load processed data for statistics
            data = np.load(output_path)
            
            tissue_ratio = data['tissue_mask'].sum() / data['tissue_mask'].size * 100
            n_classes = len(np.unique(data['y']))
            
            stats.append({
                'core': position,
                'status': 'processed',
                'tissue_percent': tissue_ratio,
                'shape': str(data['X'].shape),
                'n_bands': data['wn'].shape[0],
                'n_classes': n_classes,
                'n_pixels': data['X'].shape[0] * data['X'].shape[1],
                'n_tissue_pixels': int(data['tissue_mask'].sum()),
                'classes': np.unique(data['y']).tolist()
            })
            
            # Plot before/after comparison for first few cores
            if plot_comparison:
                try:
                    plot_spectra_comparison(
                        before_data, data['X'],
                        wn_before, data['wn'],
                        position, npz_dir
                    )
                except Exception as e:
                    print(f"Warning: Could not plot comparison for {position}: {e}")
        else:
            failed += 1
            stats.append({
                'core': position,
                'status': 'failed',
                'reason': 'processing error'
            })
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY STATISTICS")
    print("=" * 70)
    
    summary = generate_summary_statistics(stats, npz_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Processed: {processed} cores")
    print(f"⏭️  Skipped (no annotation): {skipped_no_anno} cores")
    print(f"⏭️  Skipped (empty annotation): {skipped_empty_anno} cores")
    print(f"❌ Failed: {failed} cores")
    
    if processed > 0:
        print(f"\n📊 Tissue Retention:")
        print(f"  Average: {summary['avg_tissue_percent']:.1f}% (was 15% with old method)")
        print(f"  Improvement: ~{summary['avg_tissue_percent']/15:.1f}x more training data")
        print(f"  Range: {summary['min_tissue_percent']:.1f}% - {summary['max_tissue_percent']:.1f}%")
        
        print(f"\n💾 Data Volume:")
        print(f"  Total pixels: {summary['total_pixels']:,}")
        print(f"  Tissue pixels: {summary['total_tissue_pixels']:,}")
        print(f"  Tissue ratio: {summary['total_tissue_pixels']/summary['total_pixels']*100:.1f}%")
    
    print(f"\n📁 Output directory: {npz_dir}")


if __name__ == "__main__":
    main()