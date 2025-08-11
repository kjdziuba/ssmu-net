"""
Preprocessing pipeline for hyperspectral data
Ferguson Order-A: Dual normalization approach
NO RMieS-EMSC for QCL; robustness via derivative + normalization
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


# Fixed label mapping
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
    # Extract RGB components
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


def apply_tissue_mask(data: np.ndarray, wn: np.ndarray, 
                     band: Tuple[float, float],
                     method: str = "quantile",
                     quantile: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    """Apply tissue QC mask based on spectral band with robust thresholding"""
    # Calculate area under curve in specified band
    idx = np.where((wn >= band[0]) & (wn <= band[1]))[0]
    area = data[..., idx].sum(-1)
    
    # Determine threshold
    if method == "quantile":
        thr = np.quantile(area.ravel(), quantile)
    else:
        raise ValueError("QC mask method must be 'quantile' at this stage")
    
    # Create binary mask
    mask = (area > thr).astype(np.uint8)
    
    # Apply mask to data
    data = data.copy()
    data[mask == 0] = 0
    
    return data, mask


def crop_spectral_range(data: np.ndarray, wn: np.ndarray, 
                       range_cm1: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Crop data to specified wavenumber range"""
    idx = np.where((wn >= range_cm1[0]) & (wn <= range_cm1[1]))[0]
    return data[..., idx], wn[idx]


def linear_stitch_gap(data: np.ndarray, wn: np.ndarray, 
                     gap: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation to stitch over wax gap"""
    # Find gap indices
    gap_idx = np.where((wn >= gap[0]) & (wn <= gap[1]))[0]
    
    if len(gap_idx) > 0:
        # Get boundary points
        left_idx = gap_idx[0] - 1 if gap_idx[0] > 0 else 0
        right_idx = gap_idx[-1] + 1 if gap_idx[-1] < len(wn) - 1 else len(wn) - 1
        
        # Linear interpolation
        x_points = [wn[left_idx], wn[right_idx]]
        
        # Reshape for interpolation
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        
        for i in range(len(data_flat)):
            y_points = [data_flat[i, left_idx], data_flat[i, right_idx]]
            # Clamp to edge values instead of extrapolating
            interp_func = interp1d(x_points, y_points, kind='linear', 
                                 bounds_error=False,
                                 fill_value=(y_points[0], y_points[-1]))
            data_flat[i, gap_idx] = interp_func(wn[gap_idx])
        
        data = data_flat.reshape(original_shape)
    
    return data, wn


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


def normalize_snv(data: np.ndarray) -> np.ndarray:
    """Standard Normal Variate normalization"""
    # Reshape to (n_spectra, n_channels)
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    
    # SNV: (x - mean) / std per spectrum
    means = np.mean(data_flat, axis=1, keepdims=True)
    stds = np.std(data_flat, axis=1, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    data_flat = (data_flat - means) / stds
    
    return data_flat.reshape(original_shape)


def apply_savgol(data: np.ndarray, window: int = 11, polyorder: int = 2, 
                deriv: int = 1) -> np.ndarray:
    """Apply Savitzky-Golay filter with derivative"""
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Apply along spectral axis
    return savgol_filter(data, window, polyorder, deriv=deriv, axis=-1)


def z_score_normalize(data: np.ndarray, mean: Optional[np.ndarray] = None, 
                     std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dataset-level z-score normalization"""
    # Reshape to (n_spectra, n_channels)
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    
    # Calculate or use provided statistics
    if mean is None:
        mean = np.mean(data_flat, axis=0)
    if std is None:
        std = np.std(data_flat, axis=0)
        std[std == 0] = 1  # Avoid division by zero
    
    # Apply z-score
    data_flat = (data_flat - mean) / std
    
    return data_flat.reshape(original_shape), mean, std


def mean_center(data: np.ndarray) -> np.ndarray:
    """Mean-center the data"""
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    mean = np.mean(data_flat, axis=0)
    data_flat = data_flat - mean
    return data_flat.reshape(original_shape)


def resample_uniform(data: np.ndarray, wn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to uniform wavenumber grid"""
    # Create uniform grid
    wn_uniform = np.linspace(wn.min(), wn.max(), len(wn))
    
    # Reshape for interpolation
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    data_resampled = np.zeros((data_flat.shape[0], len(wn_uniform)), dtype=np.float32)
    
    # Interpolate each spectrum
    for i in range(len(data_flat)):
        # No extrapolation, clamp to edge values
        interp_func = interp1d(wn, data_flat[i], kind='linear', 
                              bounds_error=False, 
                              fill_value=(data_flat[i, 0], data_flat[i, -1]))
        data_resampled[i] = interp_func(wn_uniform)
    
    return data_resampled.reshape(original_shape), wn_uniform


def preprocess_order_a(data: np.ndarray, wn: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Ferguson Order-A: Dual normalization approach
    Normalize BEFORE derivative; derivative AFTER smoothing; 
    re-normalize AFTER derivative; dataset z-score.
    Order per Ferguson 2022; Baker 2014.
    """
    
    # Convert to float32 for memory efficiency
    data = data.astype(np.float32)
    wn = wn.astype(np.float32)
    
    # 1. QC tissue mask
    qc_cfg = cfg['preprocess']['qc_mask']
    data, qc_mask = apply_tissue_mask(data, wn, 
                                      band=tuple(qc_cfg['band']),
                                      method=qc_cfg['method'],
                                      quantile=qc_cfg.get('quantile', 0.85))
    
    # 2. Crop to spectral range
    data, wn = crop_spectral_range(data, wn, tuple(cfg['data']['spectral_range']))
    
    # 3. Linear stitch wax gap
    data, wn = linear_stitch_gap(data, wn, tuple(cfg['data']['wax_gap']))
    
    # 4. Uniform resampling if needed
    if cfg['preprocess']['resample_uniform']:
        data, wn = resample_uniform(data, wn)
        # Assert uniform and compute delta
        diffs = np.diff(wn)
        assert np.allclose(diffs, diffs[0], rtol=1e-6), \
            f"Grid not uniform after resampling: delta range [{diffs.min()}, {diffs.max()}]"
        delta_cm1 = float(diffs[0])
        print(f"Uniform grid: {len(wn)} points, delta={delta_cm1:.3f} cm⁻¹")
    else:
        delta_cm1 = float(np.diff(wn).mean())
    
    # 5. Pre-derivative normalization
    if cfg['preprocess']['pre_norm']['type'] == 'l2':
        data = normalize_l2(data)
    elif cfg['preprocess']['pre_norm']['type'] == 'snv':
        data = normalize_snv(data)
    
    # 6. Savitzky-Golay with derivative
    data = apply_savgol(data, 
                       window=cfg['preprocess']['sg']['window'],
                       polyorder=cfg['preprocess']['sg']['polyorder'],
                       deriv=cfg['preprocess']['sg']['deriv'])
    
    # 7. Post-derivative normalization
    if cfg['preprocess']['post_norm']['type'] == 'l2':
        data = normalize_l2(data)
    elif cfg['preprocess']['post_norm']['type'] == 'snv':
        data = normalize_snv(data)
    
    # 8. Dataset z-score (will be applied with training stats during training)
    # Here we just return the data
    
    # 9. Optional mean-centering
    if cfg['preprocess']['mean_center']['enabled']:
        data = mean_center(data)
    
    return data, wn, delta_cm1, qc_mask


def preprocess_order_b(data: np.ndarray, wn: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Order-B ablation: derivative before first normalization
    QC → crop → stitch → derivative → L2 norm → z-score
    """
    
    # Convert to float32
    data = data.astype(np.float32)
    wn = wn.astype(np.float32)
    
    # 1. QC tissue mask
    qc_cfg = cfg['preprocess']['qc_mask']
    data, qc_mask = apply_tissue_mask(data, wn,
                                      band=tuple(qc_cfg['band']),
                                      method=qc_cfg['method'],
                                      quantile=qc_cfg.get('quantile', 0.85))
    
    # 2. Crop to spectral range
    data, wn = crop_spectral_range(data, wn, tuple(cfg['data']['spectral_range']))
    
    # 3. Linear stitch wax gap
    data, wn = linear_stitch_gap(data, wn, tuple(cfg['data']['wax_gap']))
    
    # 4. Uniform resampling if needed
    if cfg['preprocess']['resample_uniform']:
        data, wn = resample_uniform(data, wn)
        diffs = np.diff(wn)
        assert np.allclose(diffs, diffs[0], rtol=1e-6), \
            f"Grid not uniform after resampling"
        delta_cm1 = float(diffs[0])
    else:
        delta_cm1 = float(np.diff(wn).mean())
    
    # 5. Derivative FIRST (key difference)
    data = apply_savgol(data,
                       window=cfg['preprocess']['sg']['window'],
                       polyorder=cfg['preprocess']['sg']['polyorder'],
                       deriv=cfg['preprocess']['sg']['deriv'])
    
    # 6. Normalization AFTER derivative
    if cfg['preprocess']['post_norm']['type'] == 'l2':
        data = normalize_l2(data)
    elif cfg['preprocess']['post_norm']['type'] == 'snv':
        data = normalize_snv(data)
    
    # 7. Optional mean-centering
    if cfg['preprocess']['mean_center']['enabled']:
        data = mean_center(data)
    
    return data, wn, delta_cm1, qc_mask


def preprocess_core(core_path: str, annotation_path: str, cfg: Dict[str, Any],
                   logs_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, Dict]:
    """Preprocess a single core and its annotation"""
    
    # Load spectroscopic data
    sd = SpectroscopicData(file_path=core_path)
    data = sd.data.reshape(sd.ypixels, sd.xpixels, -1)  # (H, W, C)
    wn = sd.wavenumbers
    
    # Apply preprocessing
    if cfg['preprocess']['order'] == 'A':
        data, wn, delta_cm1, qc_mask = preprocess_order_a(data, wn, cfg)
    elif cfg['preprocess']['order'] == 'B':
        data, wn, delta_cm1, qc_mask = preprocess_order_b(data, wn, cfg)
    else:
        raise ValueError(f"Unknown preprocessing order: {cfg['preprocess']['order']}")
    
    # Load and process annotation mask
    mask = rgba_png_to_mask(annotation_path, LABEL_MAP, logs_dir)
    
    # Ensure mask matches data dimensions
    assert mask.shape[:2] == data.shape[:2], \
        f"Mask shape {mask.shape} doesn't match data shape {data.shape[:2]}"
    
    # Extract metadata (to be filled from Excel)
    metadata = {
        'core_path': core_path,
        'annotation_path': annotation_path,
        'preprocessing_order': cfg['preprocess']['order'],
        'delta_cm1': delta_cm1
    }
    
    return data, mask, wn, delta_cm1, qc_mask, metadata