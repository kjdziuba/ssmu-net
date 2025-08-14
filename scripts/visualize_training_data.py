#!/usr/bin/env python3
"""
Visualize training data exactly as it goes into the model.
Shows 256x256 center crops with annotation overlays.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from typing import Tuple, Optional, Dict, List
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Class colors matching the annotation scheme (with alpha for overlay)
CLASS_COLORS = {
    0: (128, 128, 128, 0),     # Background (transparent in overlay)
    1: (255, 0, 0, 128),       # Epithelium (red)
    2: (0, 255, 0, 128),       # Necrosis (green)  
    3: (0, 0, 255, 128),       # Blood (blue)
    4: (255, 255, 0, 128),     # Other/Artifact (yellow)
    5: (255, 0, 255, 128),     # Stroma (magenta)
    6: (0, 255, 255, 128),     # Immune/Lymphocytes (cyan)
    7: (255, 128, 0, 128),     # Nerve (orange)
}

CLASS_COLORS_SOLID = {
    0: (128, 128, 128),  # Background (gray)
    1: (255, 0, 0),      # Epithelium (red)
    2: (0, 255, 0),      # Necrosis (green)  
    3: (0, 0, 255),      # Blood (blue)
    4: (255, 255, 0),    # Other/Artifact (yellow)
    5: (255, 0, 255),    # Stroma (magenta)
    6: (0, 255, 255),    # Immune/Lymphocytes (cyan)
    7: (255, 128, 0),    # Nerve (orange)
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Epithelium',
    2: 'Necrosis',
    3: 'Blood',
    4: 'Other/Artifact',
    5: 'Stroma',
    6: 'Immune/Lymph',
    7: 'Nerve',
}

def load_npz_with_center_crop(npz_path: str, crop_size: int = 256) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load NPZ file and apply center crop as done during training"""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']  # Shape: (H, W, B)
    y = data['y']  # Shape: (H, W)
    
    # Apply center crop (same as training)
    H, W = X.shape[:2]
    if crop_size > 0 and (H > crop_size or W > crop_size):
        start_h = (H - crop_size) // 2
        start_w = (W - crop_size) // 2
        X = X[start_h:start_h+crop_size, start_w:start_w+crop_size]
        y = y[start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    # Get metadata
    meta = {}
    if 'meta' in data:
        meta_item = data['meta'].item() if hasattr(data['meta'], 'item') else data['meta']
        if isinstance(meta_item, dict):
            meta = meta_item
        elif isinstance(meta_item, str):
            try:
                meta = json.loads(meta_item)
            except:
                meta = {'core_id': Path(npz_path).stem}
    
    return X, y, meta

def create_tissue_visualization(X: np.ndarray, npz_path: str = None) -> np.ndarray:
    """Create tissue visualization using Amide I band (1600-1700 cm⁻¹)
    
    This uses the same spectral region used in preprocessing for tissue detection.
    Amide I band represents protein content (α-helical structures).
    """
    # Load wavenumbers if available
    if npz_path:
        data = np.load(npz_path, allow_pickle=True)
        if 'wn' in data:
            wn = data['wn']
        else:
            # Estimate wavenumbers (952-1800 cm⁻¹ with 425 bands)
            wn = np.linspace(952, 1800, X.shape[2])
    else:
        # Estimate wavenumbers
        wn = np.linspace(952, 1800, X.shape[2])
    
    # Find Amide I region (1600-1700 cm⁻¹)
    amide_i_mask = (wn >= 1600) & (wn <= 1700)
    amide_i_indices = np.where(amide_i_mask)[0]
    
    if len(amide_i_indices) == 0:
        # Fallback if wavenumber range doesn't include Amide I
        print("Warning: Amide I band not found, using middle bands")
        mid = X.shape[2] // 2
        amide_i_indices = range(max(0, mid-25), min(X.shape[2], mid+25))
    
    # Calculate integrated intensity in Amide I region
    amide_i_intensity = X[:, :, amide_i_indices].mean(axis=2)
    
    # Also get Amide II (1500-1600 cm⁻¹) and Lipid (1730-1750 cm⁻¹) for RGB channels
    amide_ii_mask = (wn >= 1500) & (wn <= 1600)
    amide_ii_indices = np.where(amide_ii_mask)[0]
    if len(amide_ii_indices) > 0:
        amide_ii_intensity = X[:, :, amide_ii_indices].mean(axis=2)
    else:
        amide_ii_intensity = amide_i_intensity  # Fallback
    
    lipid_mask = (wn >= 1730) & (wn <= 1750)
    lipid_indices = np.where(lipid_mask)[0]
    if len(lipid_indices) > 0:
        lipid_intensity = X[:, :, lipid_indices].mean(axis=2)
    else:
        # Use a different region if lipid band not available
        lipid_intensity = X[:, :, min(400, X.shape[2]-1)]
    
    # Create false-color image:
    # Red = Amide I (proteins)
    # Green = Amide II (protein secondary structure)
    # Blue = Lipids (cell membranes)
    rgb = np.stack([amide_i_intensity, amide_ii_intensity, lipid_intensity], axis=-1)
    
    # Normalize each channel independently for better contrast
    for i in range(3):
        channel = rgb[:, :, i]
        # Robust normalization using percentiles
        p5, p95 = np.percentile(channel[channel > 0], [5, 95]) if (channel > 0).any() else (0, 1)
        rgb[:, :, i] = np.clip((channel - p5) / (p95 - p5 + 1e-8), 0, 1)
    
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def create_mask_overlay(tissue_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create semi-transparent overlay of mask on tissue"""
    # Create RGBA image for overlay
    overlay = Image.fromarray(tissue_rgb).convert('RGBA')
    mask_layer = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
    
    # Draw mask with transparency
    mask_array = np.array(mask_layer)
    for class_id, color in CLASS_COLORS.items():
        if class_id == 0:  # Skip background
            continue
        mask_pixels = mask == class_id
        mask_array[mask_pixels] = color
    
    mask_layer = Image.fromarray(mask_array)
    
    # Composite the images
    result = Image.alpha_composite(overlay, mask_layer)
    return np.array(result.convert('RGB'))

def create_mask_boundaries(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Create boundary lines for mask classes"""
    from scipy import ndimage
    
    boundaries = np.zeros_like(mask, dtype=bool)
    for class_id in np.unique(mask):
        if class_id == 0:  # Skip background
            continue
        class_mask = mask == class_id
        # Find boundaries using morphological operations
        dilated = ndimage.binary_dilation(class_mask, iterations=thickness)
        eroded = ndimage.binary_erosion(class_mask, iterations=thickness)
        boundary = dilated ^ eroded
        boundaries |= boundary
    
    return boundaries

def extract_training_patches(X: np.ndarray, y: np.ndarray, patch_size: int = 64, 
                           n_patches: int = 4, min_foreground: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract sample training patches from the center crop"""
    H, W = X.shape[:2]
    patches = []
    
    # Try to get diverse patches
    attempts = 0
    max_attempts = n_patches * 10
    
    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        
        # Random patch location
        if H > patch_size and W > patch_size:
            start_h = np.random.randint(0, H - patch_size)
            start_w = np.random.randint(0, W - patch_size)
        else:
            start_h = start_w = 0
            
        patch_X = X[start_h:start_h+patch_size, start_w:start_w+patch_size]
        patch_y = y[start_h:start_h+patch_size, start_w:start_w+patch_size]
        
        # Check if patch has enough foreground
        foreground_ratio = (patch_y != 0).sum() / (patch_size * patch_size)
        if foreground_ratio >= min_foreground:
            patches.append((patch_X, patch_y))
    
    return patches

def visualize_core(core_name: str, X: np.ndarray, y: np.ndarray, 
                   output_dir: Path, npz_path: str = None, save_patches: bool = True) -> Dict:
    """Create comprehensive visualization for a single core"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Training Data Visualization: {core_name} (256×256 center crop)', 
                fontsize=16, fontweight='bold')
    
    # 1. Tissue visualization using Amide I
    ax1 = fig.add_subplot(gs[0, 0])
    rgb = create_tissue_visualization(X, npz_path)
    ax1.imshow(rgb)
    ax1.set_title('Tissue (Amide I/II + Lipids)')
    ax1.axis('off')
    
    # 2. Annotation Mask
    ax2 = fig.add_subplot(gs[0, 1])
    mask_vis = np.zeros((*y.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS_SOLID.items():
        mask_vis[y == class_id] = color
    ax2.imshow(mask_vis)
    ax2.set_title(f'Annotations (Classes: {np.unique(y).tolist()})')
    ax2.axis('off')
    
    # 3. Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = create_mask_overlay(rgb, y)
    ax3.imshow(overlay)
    ax3.set_title('Tissue + Annotation Overlay')
    ax3.axis('off')
    
    # 4. Boundaries
    ax4 = fig.add_subplot(gs[0, 3])
    boundaries = create_mask_boundaries(y)
    boundary_vis = rgb.copy()
    boundary_vis[boundaries] = [255, 255, 0]  # Yellow boundaries
    ax4.imshow(boundary_vis)
    ax4.set_title('Class Boundaries')
    ax4.axis('off')
    
    # 5. Class distribution bar chart
    ax5 = fig.add_subplot(gs[1, :2])
    unique, counts = np.unique(y, return_counts=True)
    colors = [np.array(CLASS_COLORS_SOLID[c])/255 for c in unique]
    bars = ax5.bar([CLASS_NAMES[c] for c in unique], counts, color=colors)
    ax5.set_title('Class Distribution (pixels)')
    ax5.set_xlabel('Class')
    ax5.set_ylabel('Pixel Count')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    total_pixels = counts.sum()
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{count/total_pixels*100:.1f}%',
                ha='center', va='bottom')
    
    # 6-9. Sample training patches (64x64)
    patches = extract_training_patches(X, y, patch_size=64, n_patches=4)
    
    for i, (patch_X, patch_y) in enumerate(patches):
        ax = fig.add_subplot(gs[2, i])
        patch_rgb = create_tissue_visualization(patch_X)  # Use tissue viz, no npz_path for patches
        patch_overlay = create_mask_overlay(patch_rgb, patch_y)
        ax.imshow(patch_overlay)
        
        # Calculate patch statistics
        fg_ratio = (patch_y != 0).sum() / (patch_y.size) * 100
        n_classes = len(np.unique(patch_y))
        
        ax.set_title(f'Training Patch {i+1} (64×64)\nFG: {fg_ratio:.1f}%, Classes: {n_classes}')
        ax.axis('off')
    
    # Statistics text
    ax_stats = fig.add_subplot(gs[1, 2:])
    ax_stats.axis('off')
    
    stats_text = f"""Core Statistics:
    • Shape: {X.shape[0]}×{X.shape[1]} pixels
    • Spectral bands: {X.shape[2]}
    • Total pixels: {y.size:,}
    • Non-background: {(y != 0).sum():,} ({(y != 0).sum()/y.size*100:.1f}%)
    • Number of classes: {len(unique)}
    • Most common: {CLASS_NAMES[unique[np.argmax(counts)]]}
    • Least common: {CLASS_NAMES[unique[np.argmin(counts)]]}
    """
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=11, 
                 verticalalignment='center', family='monospace')
    
    # Save figure
    output_path = output_dir / f"{core_name}_training_vis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return statistics for summary
    return {
        'core_name': core_name,
        'shape': X.shape,
        'n_pixels': y.size,
        'n_foreground': int((y != 0).sum()),
        'foreground_ratio': float((y != 0).sum() / y.size),
        'classes_present': unique.tolist(),
        'class_counts': {int(c): int(cnt) for c, cnt in zip(unique, counts)}
    }

def generate_html_report(stats_list: List[Dict], output_dir: Path):
    """Generate HTML report with all visualizations"""
    
    from datetime import datetime
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Training Data Visualization Report</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{ 
            color: #333; 
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .core-section {{
            background: white;
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .core-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .stats {{
            font-size: 14px;
            color: #666;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 10px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .color-box {{
            width: 20px;
            height: 20px;
            border: 1px solid #333;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>Training Data Visualization Report</h1>
    <p><em>Generated: {}</em></p>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Summary statistics
    total_cores = len(stats_list)
    total_pixels = sum(s['n_pixels'] for s in stats_list)
    total_foreground = sum(s['n_foreground'] for s in stats_list)
    avg_foreground = total_foreground / total_pixels * 100
    
    # Class distribution across all cores
    global_class_counts = {}
    for stats in stats_list:
        for class_id, count in stats['class_counts'].items():
            global_class_counts[class_id] = global_class_counts.get(class_id, 0) + count
    
    html += f"""
        <div class="summary">
            <h2>Summary</h2>
            <ul>
                <li><strong>Total cores:</strong> {total_cores}</li>
                <li><strong>Center crop size:</strong> 256×256 pixels</li>
                <li><strong>Training patch size:</strong> 64×64 pixels</li>
                <li><strong>Total pixels processed:</strong> {total_pixels:,}</li>
                <li><strong>Average foreground ratio:</strong> {avg_foreground:.2f}%</li>
                <li><strong>Classes found:</strong> {sorted(global_class_counts.keys())}</li>
            </ul>
            
            <h3>Global Class Distribution</h3>
            <div class="legend">
    """
    
    # Add legend
    for class_id in sorted(global_class_counts.keys()):
        color = CLASS_COLORS_SOLID[class_id]
        count = global_class_counts[class_id]
        percentage = count / total_pixels * 100
        html += f"""
                <div class="legend-item">
                    <div class="color-box" style="background-color: rgb{color};"></div>
                    <span>{CLASS_NAMES[class_id]}: {percentage:.2f}%</span>
                </div>
        """
    
    html += """
            </div>
        </div>
    """
    
    # Add warning if lots of background
    if avg_foreground < 50:
        html += f"""
        <div class="warning">
            <strong>⚠️ Warning:</strong> Average foreground ratio is {avg_foreground:.1f}%. 
            Consider adjusting min_foreground_ratio in training config to filter patches with too much background.
        </div>
        """
    
    # Individual cores
    html += "<h2>Individual Core Visualizations</h2>"
    
    for stats in sorted(stats_list, key=lambda x: x['core_name']):
        core_name = stats['core_name']
        img_path = f"{core_name}_training_vis.png"
        
        html += f"""
        <div class="core-section">
            <div class="core-header">
                <h3>{core_name}</h3>
                <div class="stats">
                    Shape: {stats['shape'][0]}×{stats['shape'][1]}×{stats['shape'][2]} | 
                    Foreground: {stats['foreground_ratio']*100:.1f}% | 
                    Classes: {stats['classes_present']}
                </div>
            </div>
            <img src="{img_path}" alt="{core_name} visualization">
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    # Save HTML
    report_path = output_dir / "training_data_report.html"
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize training data with annotations")
    parser.add_argument('--npz_dir', type=str, default='outputs/npz',
                       help='Directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default='outputs/training_visualization',
                       help='Output directory for visualizations')
    parser.add_argument('--max_cores', type=int, default=None,
                       help='Maximum number of cores to process')
    parser.add_argument('--crop_size', type=int, default=256,
                       help='Center crop size (default: 256)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NPZ files
    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "core_*.npz")))
    
    if args.max_cores:
        npz_files = npz_files[:args.max_cores]
    
    print(f"Found {len(npz_files)} NPZ files")
    print(f"Output directory: {output_dir}")
    print(f"Center crop size: {args.crop_size}×{args.crop_size}")
    
    # Process each core
    stats_list = []
    for npz_path in tqdm(npz_files, desc="Processing cores"):
        core_name = Path(npz_path).stem
        
        # Load data with center crop
        X, y, meta = load_npz_with_center_crop(npz_path, crop_size=args.crop_size)
        
        # Create visualization
        stats = visualize_core(core_name, X, y, output_dir, npz_path=npz_path)
        stats_list.append(stats)
    
    # Generate HTML report
    generate_html_report(stats_list, output_dir)
    
    # Save statistics JSON
    stats_path = output_dir / "training_data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_list, f, indent=2)
    
    print(f"\nVisualization complete!")
    print(f"Individual images: {output_dir}/*.png")
    print(f"HTML report: {output_dir}/training_data_report.html")
    print(f"Statistics: {stats_path}")

if __name__ == "__main__":
    main()