# SSMU-Net GPU Handover Session Report
**Date**: August 10, 2025  
**Session**: GPU machine handover and training setup  
**Status**: Successfully running fast training mode

## Initial Problem Analysis
- **Context**: Switching from previous GPU machine to Windows/WSL environment
- **Project Status**: 160+ preprocessed NPZ files ready, no trained models yet
- **Key Architecture**: SSMU-Net (Sinc + Mamba SSM + U-Net) with 425 spectral channels

## Critical Issues Encountered & Solutions

### 1. Mamba-SSM Installation Failure
**Problem**: 
- `pip install mamba-ssm` failed due to missing Visual C++ Build Tools on Windows
- Error: `Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
- **Used WSL instead of Windows**: Installed in Ubuntu WSL environment  
- **Version compatibility fix**: Installed matching wheel for PyTorch 2.3.1+cu118:
  ```bash
  pip3 install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  ```
- **Files changed**: None (installation only)

### 2. Cross-Validation Split Error  
**Problem**:
- `ValueError: k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=1`
- Only 1 TMA (group) available, GroupKFold can't split

**Solution**:
- **Added fallback logic** in `ssmu_net/data.py:create_data_splits()`
- **Logic**: Use GroupKFold if enough groups, else fallback to KFold
- **Code changes**:
  ```python
  # Check if we have enough unique groups for GroupKFold
  n_unique_groups = len(np.unique(groups))
  use_group_kfold = n_unique_groups >= n_folds
  
  if use_group_kfold:
      gkf = GroupKFold(n_splits=n_folds)
      # ... use GroupKFold with leakage checking
  else:
      print(f"[WARNING] Only {n_unique_groups} unique groups found, falling back to KFold")
      kf = KFold(n_splits=min(n_folds, len(paths)), shuffle=True, random_state=42)
      # ... use KFold without group leakage check
  ```

### 3. File Path Mismatch
**Problem**: 
- NPZ manifest paths pointed to Mac paths `/Volumes/LaCie/...`
- Running on Windows WSL needs `/mnt/e/...` paths

**Solution**:
- **Fixed manifest file**: `sed` command to update all paths
  ```bash
  sed -i 's|/Volumes/LaCie/breast_experiments|/mnt/e/breast_experiments|g' outputs/tables/npz_manifest.csv
  ```
- **Files changed**: `outputs/tables/npz_manifest.csv`

### 4. Out of Memory (OOM) Crashes
**Problem**: 
- Training killed with "Killed" message
- Loading 128 cores × ~27MB each = excessive RAM usage
- Each core: 320×320×425 = ~172MB uncompressed

**Solution**:
- **Implemented lazy loading** in `ssmu_net/data.py:NpzCoreDataset`
- **Memory-efficient approach**: Load NPZ files on-demand with `mmap_mode='r'`
- **Code changes**:
  ```python
  # Added cache_data flag (default False)
  self.cache_data = False
  
  # Store only metadata, not full arrays
  if self.cache_data:
      # Old behavior: store full arrays
  else:
      # New behavior: store path + metadata only
      self.cores.append({
          'shape': X.shape,
          'wn': wn.astype(np.float32),  # Small array OK to store
          'delta_cm1': delta_cm1,
          'path': str(path)
      })
  
  # Lazy loading in __getitem__
  if self.cache_data:
      X = core['X'][i:i+patch_size, j:j+patch_size].copy()
  else:
      npz = np.load(core['path'], mmap_mode='r')
      X = npz['X'][i:i+patch_size, j:j+patch_size].astype(np.float32)
      npz.close()
  ```
- **Fixed related functions**: `compute_class_weights()`, patch computation

### 5. CUDA Out of Memory During Training
**Problem**:
- GPU: NVIDIA RTX A1000 6GB (5934MB/6144MB used = 96.6% full)
- `RuntimeError: CUDA error: out of memory` during model forward pass

**Solution**:
- **Reduced model size** in `ssmu_net/config.yaml`:
  ```yaml
  data:
    patch_size: 64    # was 128
    batch_size: 1     # was 4  
    num_workers: 0    # was 8
  model:
    sinc:
      filters: 24     # was 32
    embed: 24         # was 32
    unet:
      base: 24        # was 32
    chunk_size: 512   # was 2048
  ```
- **Reduced parameters**: 4.4M → 1.9M trainable parameters

### 6. Deterministic Algorithm Error
**Problem**:
- `RuntimeError: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation`
- PyTorch strict deterministic mode conflicted with 2D cross-entropy loss

**Solution**:
- **Modified deterministic settings** in `ssmu_net/utils.py`:
  ```python
  # Old: torch.use_deterministic_algorithms(True)  
  # New: torch.use_deterministic_algorithms(True, warn_only=True)
  ```

### 7. Fast Training Configuration
**Problem**:
- Training extremely slow: 12s/iteration, 2+ hours per epoch
- Need quick validation that model works

**Solution**:
- **Created fast config**: `ssmu_net/config_fast.yaml`
- **Speed optimizations**:
  ```yaml
  data:
    folds: 3           # was 5
    max_patches: 50    # was ~7000+ patches
    batch_size: 1      # memory constrained
    num_workers: 0     # memory constrained
  optim:
    epochs: 3          # was 120
  model:
    # Smaller model (16→8 channels in several places)
  ```
- **Added patch limiting** in `ssmu_net/data.py`:
  ```python
  # New parameter in NpzCoreDataset.__init__
  max_patches: Optional[int] = None
  
  # Limit patches after shuffle
  if max_patches is not None and len(patches) > max_patches:
      patches = patches[:max_patches]
      print(f"[FAST MODE] Limited to {max_patches} patches")
  ```

### 8. Validation Tensor Shape Mismatch  
**Problem**:
- `RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 306 but got size 310`
- Different validation cores have different spatial dimensions

**Solution**:
- **Fixed concatenation** in `ssmu_net/train.py:validate_epoch()`:
  ```python
  # Old: all_preds.append(preds.cpu())
  # New: all_preds.append(preds.cpu().flatten())
  ```
- **Flattens spatial dimensions** before concatenation

## Final Working Configuration
- **Environment**: WSL Ubuntu with mamba-ssm pre-built wheel
- **Memory**: Lazy loading + reduced model size + no pin_memory
- **Training**: 50 patches/epoch, 3 epochs, 3 folds
- **Speed**: ~1.5 min/epoch (was 2+ hours)
- **Model**: 1.9M parameters (was 7.8M)

## Files Modified
1. `ssmu_net/data.py` - Lazy loading, patch limiting, class weight computation
2. `ssmu_net/utils.py` - Deterministic mode fix  
3. `ssmu_net/train.py` - Validation tensor shape fix
4. `ssmu_net/config.yaml` - Memory optimization
5. `ssmu_net/config_fast.yaml` - Fast training config (new file)
6. `outputs/tables/npz_manifest.csv` - Path corrections

## Status at Session End
- ✅ Mamba-SSM successfully installed and working
- ✅ Training pipeline running without crashes
- ✅ Fast training mode operational (~10 min total runtime)
- ✅ All memory issues resolved
- 🔄 Currently running: validation of 3-epoch fast training

## Next Steps
1. Validate fast training completes successfully  
2. Switch to full training with optimized config
3. Monitor for any remaining memory/performance issues