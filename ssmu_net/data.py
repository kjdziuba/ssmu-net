"""
NPZ-based dataset for hyperspectral tissue segmentation
Supports patch-based training and full-core evaluation
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import random


def resolve_npz_path(path) -> str:
    """Resolve NPZ path to work with current environment"""
    # Convert path to string if it's a Path object
    path_str = str(path)
    
    # Convert absolute paths from other systems to current working directory
    if path_str.startswith('/mnt/e/breast_experiments/ssmu_net_project/'):
        # Strip the old absolute path and use current working directory
        relative_path = path_str.replace('/mnt/e/breast_experiments/ssmu_net_project/', '')
        return relative_path
    return path_str


def compute_zscore_stats(npz_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute channel-wise mean/std over tissue pixels from training set"""
    s = None
    ss = None
    n = 0
    
    for p in npz_paths:
        resolved_path = resolve_npz_path(p)
        d = np.load(resolved_path)
        X = d['X'].astype(np.float32)
        t = d['tissue_mask'].astype(bool)
        
        # Immediately close the npz file to free memory
        d.close()
        
        if not t.any():
            continue
            
        Xf = X[t]  # (N_tissue, C)
        
        # Free the full arrays
        del X, t
        
        if s is None:
            s = Xf.sum(0)
            ss = (Xf**2).sum(0)
        else:
            s += Xf.sum(0)
            ss += (Xf**2).sum(0)
        
        n += Xf.shape[0]
    
    mean = s / max(n, 1)
    var = ss / max(n, 1) - mean**2
    std = np.sqrt(np.clip(var, 1e-8, None))
    
    return mean.astype(np.float32), std.astype(np.float32)


class NpzCoreDataset(Dataset):
    """Dataset for loading preprocessed NPZ cores"""
    
    def __init__(self, 
                 npz_paths: List[str],
                 patch_size: int = 128,
                 mode: str = 'train',
                 augment: bool = True,
                 ignore_index: int = 0,
                 z_mean: Optional[np.ndarray] = None,
                 z_std: Optional[np.ndarray] = None,
                 center_crop: Optional[int] = None,
                 max_patches: Optional[int] = None,
                 max_patches_per_core: Optional[int] = None,
                 min_foreground_ratio: float = 0.1,
                 min_tissue_ratio: float = 0.1,
                 seed: int = 1337):
        """
        Args:
            npz_paths: List of paths to NPZ files
            patch_size: Size of patches for training (ignored in eval mode)
            mode: 'train', 'val', or 'test'
            augment: Apply augmentations in train mode
            ignore_index: Label to ignore in loss computation
            z_mean: Channel-wise mean for z-score normalization
            z_std: Channel-wise std for z-score normalization
            center_crop: Size to center crop cores to (e.g., 256)
            max_patches: DEPRECATED - use max_patches_per_core instead
            max_patches_per_core: Maximum patches to sample per core
            min_foreground_ratio: Minimum ratio of non-background pixels required
            min_tissue_ratio: Minimum ratio of tissue pixels required
            seed: Random seed for reproducible patch sampling
        """
        self.npz_paths = [Path(p) for p in npz_paths]
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.ignore_index = ignore_index
        self.max_patches = max_patches  # Keep for backward compatibility
        self.max_patches_per_core = max_patches_per_core
        self.min_foreground_ratio = min_foreground_ratio
        self.min_tissue_ratio = min_tissue_ratio
        self.seed = seed
        
        # Store z-score stats
        self.z_mean = None if z_mean is None else torch.from_numpy(z_mean).float()
        self.z_std = None if z_std is None else torch.from_numpy(z_std).float()
        
        # Store paths only, load on demand to save memory
        self.cores = []
        self.metadata = []
        self.cache_data = True  # Enable RAM caching
        
        for path in self.npz_paths:
            if self.cache_data:
                # Original behavior - load into memory
                npz = np.load(resolve_npz_path(path))
                X = npz['X'].astype(np.float32)  # (H, W, C)
                y = npz['y'].astype(np.int64)  # (H, W)
                wn = npz['wn'][:]  # (C,)
                tissue_mask = npz['tissue_mask'].astype(bool)  # (H, W)
                delta_cm1 = float(npz['delta_cm1'])
                
                # Parse metadata
                meta_str = npz['meta'].item() if isinstance(npz['meta'], np.ndarray) else str(npz['meta'])
                try:
                    meta = json.loads(meta_str)
                except:
                    meta = {'core_id': Path(path).stem}
                
                # Skip uniform grid validation - we have a wax gap
                # The wax gap removal creates non-uniform spacing which is expected
                # diffs = np.diff(wn)
                # assert np.allclose(diffs, delta_cm1, rtol=1e-6), \
                #     f"Non-uniform wn found in {path}; re-run preprocessing"
                
                # Apply center cropping if specified
                if self.center_crop is not None:
                    H, W = X.shape[:2]
                    crop_size = min(H, W, self.center_crop)
                    start_h = (H - crop_size) // 2
                    start_w = (W - crop_size) // 2
                    X = X[start_h:start_h+crop_size, start_w:start_w+crop_size]
                    y = y[start_h:start_h+crop_size, start_w:start_w+crop_size]
                    tissue_mask = tissue_mask[start_h:start_h+crop_size, start_w:start_w+crop_size]
            else:
                # Memory-efficient: just check dimensions
                npz = np.load(resolve_npz_path(path), mmap_mode='r')
                X = npz['X']  # Don't convert, just check shape
                y = npz['y']  # Don't load yet
                wn = npz['wn'][:]  # Small array, ok to load
                tissue_mask = npz['tissue_mask']  # Don't load yet
                delta_cm1 = float(npz['delta_cm1'])
                
                # Note: For memory-efficient mode, we'll apply cropping when actually loading patches
                
                # Parse metadata
                meta_str = npz['meta'].item() if isinstance(npz['meta'], np.ndarray) else str(npz['meta'])
                try:
                    meta = json.loads(meta_str)
                except:
                    meta = {'core_id': path.stem}
                
                # Validate uniform grid (wn is small, ok to check)
                diffs = np.diff(wn)
                assert np.allclose(diffs, delta_cm1, rtol=1e-6), \
                    f"Non-uniform wn found in {path}; re-run preprocessing"
            
            # Store core data or just metadata depending on cache_data
            if self.cache_data:
                self.cores.append({
                    'X': X.astype(np.float32),
                    'y': y,
                    'wn': wn,
                    'tissue_mask': tissue_mask,
                    'delta_cm1': delta_cm1,
                    'path': str(path)
                })
            else:
                # Just store path and minimal metadata for lazy loading
                self.cores.append({
                    'shape': X.shape,
                    'wn': wn.astype(np.float32),  # Small array, ok to store
                    'delta_cm1': delta_cm1,
                    'path': str(path)
                })
                npz.close()  # Close memory-mapped file
            
            self.metadata.append(meta)
        
        # Compute patches for training mode
        if self.mode == 'train':
            self.patches = self._compute_patches()
        
        print(f"Loaded {len(self.cores)} cores in {mode} mode")
        if self.mode == 'train':
            print(f"  Total patches: {len(self.patches)}")
    
    def _compute_patches(self) -> List[Tuple[int, int, int]]:
        """Compute valid patch locations (core_idx, row, col) with smart filtering"""
        all_patches = []
        
        for core_idx, core in enumerate(self.cores):
            core_patches = []
            
            # Load data to check patches
            if self.cache_data:
                H, W = core['y'].shape
                tissue_mask = core['tissue_mask']
                y = core['y']
            else:
                # Load to get shape, tissue mask, and labels
                npz = np.load(resolve_npz_path(core['path']), mmap_mode='r')
                H, W = npz['y'].shape
                tissue_mask = npz['tissue_mask'][:]
                y = npz['y'][:]
            
            # PIXEL_PIXEL STRATEGY: Random sampling with tissue guarantee
            # This ensures every patch contains meaningful tissue content
            attempts = 0
            max_attempts = 1000  # Prevent infinite loops
            target_patches = self.max_patches_per_core or 50  # Default to 50 like pixel_pixel
            
            np.random.seed(self.seed + core_idx)  # Deterministic but different per core
            
            while len(core_patches) < target_patches and attempts < max_attempts:
                # Random patch location (like pixel_pixel lines 46-47)
                i = np.random.randint(0, H - self.patch_size + 1)
                j = np.random.randint(0, W - self.patch_size + 1)
                
                # Extract patch regions
                patch_tissue = tissue_mask[i:i+self.patch_size, j:j+self.patch_size]
                patch_y = y[i:i+self.patch_size, j:j+self.patch_size]
                
                # PIXEL_PIXEL'S KEY REQUIREMENT: patch must contain at least one tissue pixel
                # (like pixel_pixel line 49: if (m_patch != 0).any())
                if not patch_tissue.any():
                    attempts += 1
                    continue
                
                # Additional quality checks
                tissue_ratio = patch_tissue.mean()
                if tissue_ratio < self.min_tissue_ratio:
                    attempts += 1
                    continue
                
                # Check for meaningful annotation content
                if self.min_foreground_ratio > 0:
                    tissue_pixels = patch_y[patch_tissue > 0]
                    if len(tissue_pixels) > 0:
                        foreground_ratio = (tissue_pixels > 0).mean()
                        if foreground_ratio < self.min_foreground_ratio:
                            attempts += 1
                            continue
                
                # Patch passes all checks - add it
                core_patches.append((core_idx, i, j))
                attempts += 1
            
            print(f"Core {core_idx}: {len(core_patches)} tissue-guaranteed patches (attempts: {attempts})")
            
            # Close the npz file if lazy loading
            if not self.cache_data:
                npz.close()
            
            # Apply per-core limit if specified
            if self.max_patches_per_core is not None and len(core_patches) > self.max_patches_per_core:
                # Randomly sample from this core's patches
                random.shuffle(core_patches)
                core_patches = core_patches[:self.max_patches_per_core]
                print(f"Core {core_idx}: sampled {self.max_patches_per_core} patches")
            
            all_patches.extend(core_patches)
        
        # Randomize global patch ordering
        random.shuffle(all_patches)
        
        # Apply global limit for backward compatibility
        if self.max_patches is not None and len(all_patches) > self.max_patches:
            all_patches = all_patches[:self.max_patches]
            print(f"[GLOBAL LIMIT] Limited to {self.max_patches} patches total")
        
        return all_patches
    
    def _augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial augmentations"""
        # Random flips
        if random.random() > 0.5:
            X = np.flip(X, axis=0).copy()
            y = np.flip(y, axis=0).copy()
        if random.random() > 0.5:
            X = np.flip(X, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            X = np.rot90(X, k, axes=(0, 1)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()
        
        return X, y
    
    def __len__(self) -> int:
        if self.mode == 'train':
            return len(self.patches)
        else:
            return len(self.cores)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == 'train':
            # Get patch
            core_idx, i, j = self.patches[idx]
            core = self.cores[core_idx]
            
            # Load data if not cached
            if self.cache_data:
                X = core['X'][i:i+self.patch_size, j:j+self.patch_size].copy()
                y = core['y'][i:i+self.patch_size, j:j+self.patch_size].copy()
                tissue_mask = core['tissue_mask'][i:i+self.patch_size, j:j+self.patch_size].copy()
            else:
                # Lazy load from disk
                npz = np.load(resolve_npz_path(core['path']), mmap_mode='r')
                X = npz['X'][i:i+self.patch_size, j:j+self.patch_size].astype(np.float32)
                y = npz['y'][i:i+self.patch_size, j:j+self.patch_size].astype(np.int64)
                tissue_mask = npz['tissue_mask'][i:i+self.patch_size, j:j+self.patch_size].astype(np.uint8)
                npz.close()
            
            # Apply augmentations
            if self.augment:
                X, y = self._augment(X, y)
                # Note: tissue_mask augmentation skipped as it's not used in training
            
            # Set non-tissue pixels to ignore_index
            y[tissue_mask == 0] = self.ignore_index
            
            # Safety check: ensure patch has some non-ignored pixels
            # This prevents NaN losses when all pixels are background
            if self.ignore_index == 0 and (y != self.ignore_index).sum() == 0:
                # Patch is all background - sample a different patch
                # Use modulo to wrap around if we reach the end
                return self.__getitem__((idx + 1) % len(self))
            
        else:
            # Return full core for evaluation
            core_idx = idx
            core = self.cores[idx]
            
            if self.cache_data:
                X = core['X'].copy()
                y = core['y'].copy()
                tissue_mask = core['tissue_mask'].copy()
            else:
                # Lazy load from disk
                npz = np.load(resolve_npz_path(core['path']), mmap_mode='r')
                X = npz['X'][:].astype(np.float32)
                y = npz['y'][:].astype(np.int64)
                tissue_mask = npz['tissue_mask'][:].astype(np.uint8)
                npz.close()
            
            # Set non-tissue pixels to ignore_index
            y[tissue_mask == 0] = self.ignore_index
        
        # Apply z-score normalization
        if self.z_mean is not None and self.z_std is not None:
            # X is (H, W, C), z_mean and z_std are (C,)
            X = (X - self.z_mean.numpy()) / self.z_std.numpy()
        
        # Convert to tensors
        # X: (H, W, C) -> (C, H, W) for CNN
        X = torch.from_numpy(X).float().permute(2, 0, 1)
        y = torch.from_numpy(y).long()
        
        # Get wavenumbers (same for all cores, so load from first if needed)
        if hasattr(self, 'wn'):
            wn = self.wn
        elif 'wn' in self.cores[0]:
            wn = torch.from_numpy(self.cores[0]['wn']).float()
            self.wn = wn  # Cache for future use
        else:
            # Load from first core's NPZ file
            npz = np.load(resolve_npz_path(self.cores[0]['path']), mmap_mode='r')
            wn = torch.from_numpy(npz['wn'][:].astype(np.float32))
            npz.close()
            self.wn = wn  # Cache for future use
        
        # Get core info
        core_id = Path(self.cores[core_idx]['path']).stem
        
        return {
            'X': X,
            'y': y,
            'wn': wn,
            'idx': idx,
            'core_idx': core_idx,
            'core_id': core_id
        }


def create_data_splits(manifest_path: str, 
                      n_folds: int = 5,
                      holdout_keys: Optional[List[str]] = None,
                      seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Create cross-validation splits ensuring no data leakage
    
    Args:
        manifest_path: Path to npz_manifest.csv
        n_folds: Number of CV folds
        holdout_keys: Optional list of columns to group by (e.g., ['slide_id', 'block_id'])
        seed: Random seed
    
    Returns:
        List of fold dictionaries with 'train' and 'val' NPZ paths
    """
    # Load manifest
    df = pd.read_csv(manifest_path)
    
    # Create composite groups from holdout keys
    if holdout_keys:
        groups = df[holdout_keys].astype(str).agg('|'.join, axis=1).values
    else:
        # Fallback to slide_id if no holdout keys specified
        groups = df['slide_id'].astype(str).values
    
    paths = df['npz'].values
    
    # Create splits
    splits = []
    
    # Use dummy labels for splitting
    dummy_labels = np.zeros(len(df))
    
    # Check if we have enough unique groups for GroupKFold
    n_unique_groups = len(np.unique(groups))
    use_group_kfold = n_unique_groups >= n_folds
    
    if use_group_kfold:
        # Use GroupKFold when we have enough groups
        gkf = GroupKFold(n_splits=n_folds)
        split_iterator = gkf.split(paths, dummy_labels, groups)
        print(f"Using GroupKFold with {n_unique_groups} unique groups")
    else:
        # Fallback to regular KFold when not enough groups
        print(f"[WARNING] Only {n_unique_groups} unique groups found, falling back to KFold")
        kf = KFold(n_splits=min(n_folds, len(paths)), shuffle=True, random_state=42)
        split_iterator = kf.split(paths)
    
    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
        train_paths = paths[train_idx].tolist()
        val_paths = paths[val_idx].tolist()
        
        # Log split info
        if use_group_kfold:
            # Only check group overlap when using GroupKFold
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            
            print(f"Fold {fold_idx}:")
            print(f"  Train: {len(train_paths)} cores from {len(train_groups)} groups")
            print(f"  Val: {len(val_paths)} cores from {len(val_groups)} groups")
            
            # Verify no overlap only when using GroupKFold
            assert len(train_groups & val_groups) == 0, "Data leakage detected!"
        else:
            # For regular KFold, just report core counts
            print(f"Fold {fold_idx}:")
            print(f"  Train: {len(train_paths)} cores (no grouping)")
            print(f"  Val: {len(val_paths)} cores (no grouping)")
        
        splits.append({
            'fold': fold_idx,
            'train': train_paths,
            'val': val_paths
        })
    
    return splits


def compute_class_weights(dataset: NpzCoreDataset, 
                         num_classes: int = 8,
                         ignore_index: int = 0,
                         clip_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """Compute inverse frequency class weights from dataset with optional clipping
    
    Args:
        dataset: Dataset to compute weights from
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        clip_range: Optional (min, max) range to clip weights to
    
    Returns:
        Class weights tensor
    """
    counts = np.zeros(num_classes)
    
    for core in dataset.cores:
        # Check if data is cached or needs to be loaded
        if 'y' in core:
            # Data is cached
            y = core['y']
            tissue_mask = core['tissue_mask']
        else:
            # Lazy load from disk
            npz = np.load(resolve_npz_path(core['path']), mmap_mode='r')
            y = npz['y'][:]
            tissue_mask = npz['tissue_mask'][:]
            npz.close()
        
        # Only count tissue pixels
        y_tissue = y[tissue_mask > 0]
        
        for c in range(num_classes):
            counts[c] += (y_tissue == c).sum()
    
    # Compute inverse frequency weights
    counts = np.maximum(counts, 1)  # Avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    
    # Apply clipping if specified
    if clip_range is not None:
        min_weight, max_weight = clip_range
        weights = np.clip(weights, min_weight, max_weight)
        # Re-normalize after clipping
        weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(cfg: Dict[str, Any],
                      train_paths: List[str],
                      val_paths: List[str],
                      test_paths: Optional[List[str]] = None,
                      use_zscore: bool = True) -> Dict[str, DataLoader]:
    """Create dataloaders for training, validation, and optionally test
    
    Args:
        cfg: Configuration dictionary
        train_paths: Training NPZ file paths
        val_paths: Validation NPZ file paths
        test_paths: Optional test NPZ file paths
        use_zscore: If False, skip z-score normalization (use raw double-L2 features)
    """
    
    # Compute z-score statistics from training set only (if enabled)
    if use_zscore:
        print("Computing z-score statistics from training set...")
        z_mean, z_std = compute_zscore_stats(train_paths)
    else:
        print("Skipping z-score normalization (using raw double-L2 features)")
        z_mean, z_std = None, None
    
    # Save z-score stats (if computed)
    if use_zscore:
        stats_path = Path(cfg['runtime_paths']['tables']) / 'zscore_stats.csv'
        pd.DataFrame({
            'wn_idx': np.arange(z_mean.size),
            'mean': z_mean,
            'std': z_std
        }).to_csv(stats_path, index=False)
        print(f"Saved z-score stats to {stats_path}")
    else:
        print("No z-score stats to save (using raw features)")
    
    # Create datasets with z-score normalization
    train_dataset = NpzCoreDataset(
        train_paths,
        patch_size=cfg['data']['patch_size'],
        mode='train',
        augment=True,
        ignore_index=cfg['data'].get('ignore_index', 0),
        z_mean=z_mean,
        z_std=z_std,
        center_crop=cfg['data'].get('center_crop', None),
        max_patches=cfg['data'].get('max_patches', None),  # Backward compat
        max_patches_per_core=cfg['data'].get('max_patches_per_core_train', None),
        min_foreground_ratio=cfg['data'].get('min_foreground_ratio', 0.1),
        min_tissue_ratio=cfg['data'].get('min_tissue_ratio', 0.1)
    )
    
    val_dataset = NpzCoreDataset(
        val_paths,
        patch_size=cfg['data']['patch_size'],
        mode='val',
        augment=False,
        ignore_index=cfg['data'].get('ignore_index', 0),
        z_mean=z_mean,
        z_std=z_std,
        center_crop=cfg['data'].get('center_crop', None),
        max_patches_per_core=cfg['data'].get('max_patches_per_core_val', None),
        min_foreground_ratio=0.05,  # Minimal filtering to avoid all-background patches
        min_tissue_ratio=0.1  # Basic tissue check
    )
    
    # Determine if we should use persistent workers
    use_persistent = cfg['data']['num_workers'] > 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=False,
        drop_last=True,
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full cores for validation
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=False,
        persistent_workers=use_persistent
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Optionally add test loader
    if test_paths:
        test_dataset = NpzCoreDataset(
            test_paths,
            mode='test',
            augment=False,
            ignore_index=0,
            z_mean=z_mean,
            z_std=z_std,
            center_crop=cfg['data'].get('center_crop', None)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg['data']['num_workers'],
            pin_memory=False,
            persistent_workers=use_persistent
        )
        
        loaders['test'] = test_loader
    
    return loaders


def save_splits(splits: List[Dict], output_path: str) -> None:
    """Save splits to JSON for reproducibility"""
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {output_path}")


def load_splits(splits_path: str) -> List[Dict]:
    """Load splits from JSON"""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return splits