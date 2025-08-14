"""
Optimized preprocessing pipeline for hyperspectral data
Based on pixel_pixel success and domain expert advice
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from PIL import Image
from tqdm import tqdm
from specml.data.spectroscopic_data import SpectroscopicData
from .utils import save_json


# Fixed label mapping (same as original)
LABEL_MAP = {
    (0, 0, 0, 255): 0,       # Background
    (0, 255, 0, 255): 1,     # Normal Epithelium (green)
    (128, 0, 128, 255): 2,   # Normal Stroma (purple)
    (255, 0, 255, 255): 3,   # Cancer Epithelium (magenta)
    (0, 0, 255, 255): 4,     # Cancer Associated Stroma (blue)
    (255, 0, 0, 255): 5,     # Blood (red)
    (255, 165, 0, 255): 6,   # Concretions/Necrosis (orange)
    (255, 255, 0, 255): 7    # Immune Infiltration (yellow)
}


def save_label_map(npz_dir: str) -> None:
    """Save label mapping to JSON file"""
    label_map_path = Path(npz_dir) / 'label_map.json'
    with open(label_map_path, 'w') as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2)
    print(f"Label map saved to {label_map_path}")


def rgba_png_to_mask(png_path: str, label_map: Dict, logs_dir: str) -> np.ndarray:
    """Vectorized RGBA to mask conversion with unknown color detection"""
    arr = np.array(Image.open(png_path).convert('RGBA'), dtype=np.uint8)
    
    # Pack RGBA into 32-bit integers for fast comparison
    packed = (arr[..., 0].astype(np.uint32) << 24) | \
             (arr[..., 1].astype(np.uint32) << 16) | \
             (arr[..., 2].astype(np.uint32) << 8)  | \
             arr[..., 3].astype(np.uint32)
    
    # Build lookup table
    lut = {((r << 24) | (g << 16) | (b << 8) | a): cls 
           for (r, g, b, a), cls in label_map.items()}
    
    # Initialize mask with background (0) instead of 255
    mask = np.zeros(packed.shape, dtype=np.uint8)
    
    # Map known colors
    for key, cls in lut.items():
        mask[packed == key] = cls
    
    # For grayscale colors (R==G==B), treat as background
    r = (packed >> 24) & 255
    g = (packed >> 16) & 255
    b = (packed >> 8) & 255
    
    # Identify grayscale pixels (where R≈G≈B) that aren't pure black
    is_gray = (r == g) & (g == b) & (r > 0)
    
    # Count unknown non-gray colors
    unknown_mask = np.zeros_like(mask, dtype=bool)
    for key in lut.keys():
        unknown_mask |= (packed == key)
    unknown_mask = ~unknown_mask & ~is_gray  # Not known and not gray
    
    uniq, cnt = np.unique(packed[unknown_mask], return_counts=True)
    if uniq.size:
        unknown = {str(((u >> 24) & 255, (u >> 16) & 255, (u >> 8) & 255, u & 255)): int(c) 
                   for u, c in zip(uniq, cnt)}
        unknown_path = Path(logs_dir) / 'unknown_colors.json'
        with open(unknown_path, 'w') as f:
            json.dump(unknown, f, indent=2)
        print(f"Warning: {len(unknown)} unknown non-gray colors found (treating as background)")
    
    # Log grayscale pixel count if significant
    gray_count = is_gray.sum()
    if gray_count > 0:
        print(f"Info: {gray_count} grayscale pixels found (treating as background)")
    
    # Validate mask values
    assert mask.min() >= 0 and mask.max() <= 7, f"Mask values out of range: [{mask.min()}, {mask.max()}]"
    
    return mask


def apply_tissue_mask_fixed(data: np.ndarray, wn: np.ndarray, 
                           band: Tuple[float, float],
                           threshold: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply tissue QC mask based on fixed threshold (not percentile)
    Validated approach from QCL prostate data: threshold=2.5-3.0
    """
    # Calculate area under curve in specified band (Amide I)
    idx = np.where((wn >= band[0]) & (wn <= band[1]))[0]
    if len(idx) == 0:
        print(f"Warning: Band {band} not found in wavenumbers. Skipping tissue mask.")
        return data, np.ones(data.shape[:2], dtype=np.uint8)
    
    area = data[..., idx].sum(-1)
    
    # Use fixed threshold (validated on QCL data)
    mask = (area > threshold).astype(np.uint8)
    
    # Report tissue percentage
    tissue_percent = mask.sum() / mask.size * 100
    print(f"Tissue mask: {tissue_percent:.1f}% pixels retained (threshold={threshold})")
    
    # Apply mask to data
    data = data.copy()
    data[mask == 0] = 0
    
    return data, mask


def remove_wax_bands(data: np.ndarray, wn: np.ndarray, 
                     gap: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove wax gap region entirely (no interpolation)
    This removes synthetic data and reduces computational load
    """
    # Find indices outside the gap
    keep_idx = np.where((wn < gap[0]) | (wn > gap[1]))[0]
    
    if len(keep_idx) < len(wn):
        removed_bands = len(wn) - len(keep_idx)
        print(f"Removed {removed_bands} bands in wax gap {gap} ({removed_bands/len(wn)*100:.1f}%)")
    
    return data[..., keep_idx], wn[keep_idx]


def crop_spectral_range(data: np.ndarray, wn: np.ndarray, 
                       range_cm1: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Crop data to specified wavenumber range"""
    idx = np.where((wn >= range_cm1[0]) & (wn <= range_cm1[1]))[0]
    return data[..., idx], wn[idx]


def normalize_l2(data: np.ndarray) -> np.ndarray:
    """Per-spectrum L2 normalization"""
    # Reshape to (n_spectra, n_channels)
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    
    # L2 norm per spectrum
    norms = np.linalg.norm(data_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    data_flat = data_flat / norms
    
    return data_flat.reshape(original_shape)


def apply_savgol(data: np.ndarray, window: int = 11, polyorder: int = 5, 
                deriv: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay filter with derivative
    Optimized parameters: window=11, polyorder=5, deriv=2
    """
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Validate parameters
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be less than window ({window})")
    
    # Apply along spectral axis
    return savgol_filter(data, window, polyorder, deriv=deriv, axis=-1)


def preprocess_optimized(data: np.ndarray, wn: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Optimized preprocessing pipeline
    Based on pixel_pixel success (0.9 mIoU) and domain expert advice
    
    Pipeline:
    1. Tissue mask (fixed threshold=2.5)
    2. Crop spectral range
    3. Remove wax gap (no interpolation)
    4. L2 normalization
    5. Savitzky-Golay with 2nd derivative
    6. L2 normalization again
    """
    
    # Convert to float32 for memory efficiency
    data = data.astype(np.float32)
    wn = wn.astype(np.float32)
    
    # 1. QC tissue mask with fixed threshold
    qc_cfg = cfg['preprocess']['qc_mask']
    data, qc_mask = apply_tissue_mask_fixed(
        data, wn, 
        band=tuple(qc_cfg['band']),
        threshold=qc_cfg.get('threshold', 2.5)  # Fixed threshold, not percentile
    )
    
    # 2. Crop to spectral range
    data, wn = crop_spectral_range(data, wn, tuple(cfg['data']['spectral_range']))
    
    # 3. Remove wax gap (not interpolate)
    data, wn = remove_wax_bands(data, wn, tuple(cfg['data']['wax_gap']))
    
    # 4. First L2 normalization
    data = normalize_l2(data)
    print("Applied first L2 normalization")
    
    # 5. Savitzky-Golay with 2nd derivative
    data = apply_savgol(
        data, 
        window=cfg['preprocess']['sg']['window'],
        polyorder=cfg['preprocess']['sg']['polyorder'],
        deriv=cfg['preprocess']['sg']['deriv']
    )
    print(f"Applied SG filter: window={cfg['preprocess']['sg']['window']}, "
          f"order={cfg['preprocess']['sg']['polyorder']}, "
          f"deriv={cfg['preprocess']['sg']['deriv']}")
    
    # 6. Second L2 normalization (for numerical stability)
    data = normalize_l2(data)
    print("Applied second L2 normalization")
    
    # Calculate delta (should be uniform after wax removal)
    delta_cm1 = float(np.diff(wn).mean())
    print(f"Final grid: {len(wn)} points, delta={delta_cm1:.3f} cm⁻¹")
    
    return data, wn, delta_cm1, qc_mask


def process_core(
    core_path: str,
    png_path: str,
    output_path: str,
    cfg: Dict[str, Any],
    logs_dir: str,
    meta_dict: Optional[Dict] = None
) -> bool:
    """Process a single core with optimized preprocessing"""
    
    try:
        # Load SpectroscopicData
        sd = SpectroscopicData(file_path=core_path)
        
        # Get dimensions
        H, W = sd.ypixels, sd.xpixels
        
        # Reshape data to (H, W, B)
        data = sd.data.reshape(H, W, -1)
        wn = sd.wavenumbers
        
        # Apply optimized preprocessing
        data_processed, wn_processed, delta_cm1, tissue_mask = preprocess_optimized(data, wn, cfg)
        
        # Load annotation mask
        if Path(png_path).exists():
            mask = rgba_png_to_mask(png_path, LABEL_MAP, logs_dir)
            
            # Validate mask dimensions
            if mask.shape != (H, W):
                print(f"Warning: Mask shape {mask.shape} != data shape ({H}, {W})")
                return False
        else:
            print(f"Warning: No annotation found at {png_path}, using zeros")
            mask = np.zeros((H, W), dtype=np.uint8)
        
        # Create metadata
        meta = meta_dict if meta_dict else {}
        meta.update({
            'preprocessing': 'optimized',
            'tissue_threshold': cfg['preprocess']['qc_mask'].get('threshold', 2.5),
            'sg_window': cfg['preprocess']['sg']['window'],
            'sg_polyorder': cfg['preprocess']['sg']['polyorder'],
            'sg_deriv': cfg['preprocess']['sg']['deriv'],
            'double_l2_norm': True,
            'wax_removed': True
        })
        
        # Save as NPZ
        np.savez_compressed(
            output_path,
            X=data_processed.astype(np.float32),
            y=mask.astype(np.uint8),
            wn=wn_processed.astype(np.float32),
            tissue_mask=tissue_mask.astype(np.uint8),
            delta_cm1=np.array(delta_cm1, dtype=np.float32),
            meta=json.dumps(meta)
        )
        
        # Report statistics
        unique, counts = np.unique(mask, return_counts=True)
        print(f"Saved {Path(output_path).name}: shape={data_processed.shape}, "
              f"classes={unique.tolist()}, tissue={tissue_mask.sum()/tissue_mask.size*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error processing core: {e}")
        return False