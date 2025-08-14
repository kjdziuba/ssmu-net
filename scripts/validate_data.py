#!/usr/bin/env python3
"""
Data validation script to verify NPZ preprocessing and mask alignment.
Generates visual comparisons of original annotations vs processed masks.
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Class colors matching the annotation scheme
CLASS_COLORS = {
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

def load_npz_data(npz_path):
    """Load NPZ file and extract data/mask"""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']  # Shape: (H, W, B)
    y = data['y']  # Shape: (H, W)
    
    # Get metadata if available
    meta = {}
    if 'meta' in data:
        meta_item = data['meta'].item() if hasattr(data['meta'], 'item') else data['meta']
        if isinstance(meta_item, dict):
            meta = meta_item
    
    return X, y, meta

def load_png_annotation(png_path):
    """Load original PNG annotation if available"""
    if os.path.exists(png_path):
        img = Image.open(png_path)
        return np.array(img)
    return None

def create_mask_visualization(mask, title="Mask"):
    """Create colored visualization of mask"""
    H, W = mask.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        vis[mask == class_id] = color
    
    return vis

def plot_validation(core_name, X, y, png_img, output_dir):
    """Create validation plot for a single core"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Validation: {core_name}', fontsize=14, fontweight='bold')
    
    # 1. RGB composite from hyperspectral (using specific bands)
    rgb_composite = X[:, :, [50, 100, 150]] if X.shape[2] > 150 else X[:, :, :3]
    rgb_composite = (rgb_composite - rgb_composite.min()) / (rgb_composite.max() - rgb_composite.min() + 1e-8)
    axes[0, 0].imshow(rgb_composite)
    axes[0, 0].set_title('Hyperspectral RGB Composite')
    axes[0, 0].axis('off')
    
    # 2. Processed mask
    mask_vis = create_mask_visualization(y)
    axes[0, 1].imshow(mask_vis)
    axes[0, 1].set_title(f'Processed Mask (unique: {np.unique(y).tolist()})')
    axes[0, 1].axis('off')
    
    # 3. Original PNG (if available)
    if png_img is not None:
        axes[0, 2].imshow(png_img)
        axes[0, 2].set_title('Original PNG Annotation')
    else:
        axes[0, 2].text(0.5, 0.5, 'PNG not found', ha='center', va='center')
        axes[0, 2].set_title('Original PNG Annotation')
    axes[0, 2].axis('off')
    
    # 4. Mask overlay on data
    overlay = rgb_composite.copy()
    mask_overlay = create_mask_visualization(y).astype(float) / 255
    alpha = 0.4
    overlay = (1 - alpha) * overlay + alpha * mask_overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Mask Overlay on Data')
    axes[1, 0].axis('off')
    
    # 5. Class distribution
    unique, counts = np.unique(y, return_counts=True)
    total_pixels = y.size
    bars = axes[1, 1].bar(unique, counts)
    
    # Color bars by class
    for i, (class_id, count) in enumerate(zip(unique, counts)):
        color = np.array(CLASS_COLORS.get(class_id, (128, 128, 128))) / 255
        bars[i].set_color(color)
        # Add percentage labels
        pct = 100 * count / total_pixels
        axes[1, 1].text(class_id, count, f'{pct:.1f}%', 
                       ha='center', va='bottom', fontsize=8)
    
    axes[1, 1].set_xlabel('Class ID')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].set_xticks(unique)
    
    # 6. Mask statistics
    stats_text = []
    stats_text.append(f"Shape: {y.shape}")
    stats_text.append(f"Total pixels: {total_pixels:,}")
    stats_text.append(f"Non-background: {(y > 0).sum():,} ({100*(y > 0).mean():.1f}%)")
    stats_text.append("\nPer-class pixels:")
    for class_id in unique:
        count = (y == class_id).sum()
        pct = 100 * count / total_pixels
        name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        stats_text.append(f"  {name}: {count:,} ({pct:.1f}%)")
    
    # Check for issues
    if -100 in unique:
        stats_text.append("\n⚠️ WARNING: Found -100 in mask!")
    if y.max() > 7:
        stats_text.append(f"\n⚠️ WARNING: Max value {y.max()} > 7!")
    if (y > 0).sum() == 0:
        stats_text.append("\n⚠️ WARNING: No annotated pixels!")
    
    axes[1, 2].text(0.05, 0.95, '\n'.join(stats_text), 
                   transform=axes[1, 2].transAxes,
                   verticalalignment='top',
                   fontfamily='monospace',
                   fontsize=9)
    axes[1, 2].set_title('Statistics')
    axes[1, 2].axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(color=np.array(color)/255, 
                                     label=CLASS_NAMES.get(i, f'Class {i}'))
                      for i, color in CLASS_COLORS.items()]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05),
              ncol=len(CLASS_COLORS), fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{core_name}_validation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_html_report(validation_results, output_dir):
    """Generate HTML report with all validation images"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Validation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .summary { background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }
            .warning { color: #ff6600; font-weight: bold; }
            .success { color: #00aa00; font-weight: bold; }
            .core-section { margin: 30px 0; border-top: 2px solid #ccc; padding-top: 20px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .stats-table { border-collapse: collapse; width: 100%; }
            .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .stats-table th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>NPZ Data Validation Report</h1>
    """
    
    # Summary section
    total_cores = len(validation_results)
    cores_with_issues = sum(1 for r in validation_results if r['has_issues'])
    
    html_content += f"""
    <div class="summary">
        <h2>Summary</h2>
        <p>Total cores processed: <b>{total_cores}</b></p>
        <p>Cores with issues: <span class="{'warning' if cores_with_issues > 0 else 'success'}">{cores_with_issues}</span></p>
    </div>
    """
    
    # Overall statistics table
    html_content += """
    <h2>Overall Statistics</h2>
    <table class="stats-table">
        <tr>
            <th>Core</th>
            <th>Shape</th>
            <th>Non-background %</th>
            <th>Unique Classes</th>
            <th>Issues</th>
        </tr>
    """
    
    for result in validation_results:
        issues_text = ', '.join(result['issues']) if result['issues'] else 'None'
        issue_class = 'warning' if result['has_issues'] else 'success'
        html_content += f"""
        <tr>
            <td>{result['core_name']}</td>
            <td>{result['shape']}</td>
            <td>{result['non_bg_pct']:.1f}%</td>
            <td>{result['unique_classes']}</td>
            <td class="{issue_class}">{issues_text}</td>
        </tr>
        """
    
    html_content += "</table>"
    
    # Individual core sections
    html_content += "<h2>Individual Core Validations</h2>"
    
    for result in validation_results:
        html_content += f"""
        <div class="core-section">
            <h3>{result['core_name']}</h3>
            <img src="{os.path.basename(result['image_path'])}" alt="{result['core_name']} validation">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, 'validation_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def main():
    # Paths
    npz_dir = "outputs/npz"
    png_dir = "/Volumes/LaCie/Breast_Cancer_Biomax_QCL/Isolated Cores/BR2082_H260_1"
    output_dir = "outputs/validation"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NPZ files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "core_*.npz")))
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ files to validate")
    
    validation_results = []
    
    for i, npz_path in enumerate(npz_files):
        core_name = Path(npz_path).stem
        print(f"\n[{i+1}/{len(npz_files)}] Processing {core_name}")
        
        # Load NPZ data
        X, y, meta = load_npz_data(npz_path)
        
        # Try to find corresponding PNG
        png_path = os.path.join(png_dir, f"{core_name}.png")
        png_img = load_png_annotation(png_path)
        
        # Generate validation plot
        img_path = plot_validation(core_name, X, y, png_img, output_dir)
        
        # Collect statistics
        unique_classes = np.unique(y).tolist()
        non_bg_pct = 100 * (y > 0).mean()
        
        # Check for issues
        issues = []
        if -100 in unique_classes:
            issues.append("Contains -100")
        if y.max() > 7:
            issues.append(f"Max value {y.max()} > 7")
        if (y > 0).sum() == 0:
            issues.append("No annotations")
        if non_bg_pct < 1.0:
            issues.append("Very few annotations")
        
        validation_results.append({
            'core_name': core_name,
            'image_path': img_path,
            'shape': str(y.shape),
            'non_bg_pct': non_bg_pct,
            'unique_classes': str(unique_classes),
            'issues': issues,
            'has_issues': len(issues) > 0
        })
        
        print(f"  Shape: {y.shape}, Non-background: {non_bg_pct:.1f}%, Classes: {unique_classes}")
        if issues:
            print(f"  ⚠️ Issues: {', '.join(issues)}")
    
    # Generate HTML report
    report_path = generate_html_report(validation_results, output_dir)
    print(f"\n✅ Validation complete!")
    print(f"Generated {len(validation_results)} validation images")
    print(f"HTML report: {report_path}")
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, 'validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"Summary JSON: {summary_path}")

if __name__ == "__main__":
    main()