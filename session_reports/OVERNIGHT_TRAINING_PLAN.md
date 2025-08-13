# SSMU-Net Overnight Training Plan
**Date**: August 10, 2025  
**Expected Runtime**: 12-16 hours  
**Configuration**: Optimized for meaningful signal

## Launch Command
```bash
cd /mnt/e/breast_experiments/ssmu_net_project
python3 scripts/run_train.py --config ssmu_net/config_overnight.yaml
```

## Key Improvements Made

### 1. Smart Patch Sampling
- **Per-core sampling**: 8 patches per core for training (vs global 50 limit)
- **Foreground filtering**: Only patches with ≥2% non-background tissue
- **Tissue filtering**: Only patches with ≥15% tissue coverage
- **Expected patches**: ~800/epoch vs 50 previously

### 2. Model Size Increase
- **Sinc filters**: 16 → 24 learned spectral bands
- **Embedding**: 16 → 32 channels  
- **U-Net base**: 16 → 32 channels
- **Mamba layers**: 1 → 2 (better temporal modeling)
- **Pooling**: mean → attention (better spectral aggregation)

### 3. Loss Function Improvements
- **Class weights**: Re-enabled with clipping [0.25, 4.0]
- **Label smoothing**: Reduced to 0.02 (vs 0.05)
- **Training loss logging**: Fixed to show actual values

### 4. Training Optimization  
- **Epochs**: 10 → 25 per fold
- **Learning rate**: 1e-4 → 3e-4 (faster convergence)
- **Early stopping**: 7 epochs patience
- **Warmup**: 2 epochs at 0.1× learning rate

## Expected Improvements

### Performance Targets
- **Current (fast mode)**: 1.4% mIoU
- **Short term goal**: 15-20% mIoU  
- **Success threshold**: >5% mIoU consistently
- **Validation**: Better than random guessing (12.5%)

### Runtime Estimation
- **Patches per epoch**: ~800 (vs 50 previously) 
- **Iterations per epoch**: ~800 (batch_size=1)
- **Total training time**: 3 folds × 25 epochs × ~3 min/epoch = 3.75 hours minimum
- **With validation/overhead**: 12-16 hours total

### Memory Usage
- **Model size**: ~4.5M parameters (vs 1.9M fast mode)
- **GPU memory**: Should fit in 6GB RTX A1000
- **Lazy loading**: Prevents RAM overflow

## What This Proves
If successful (mIoU > 5%), this demonstrates:

1. **Architecture works**: Mamba SSM effectively processes hyperspectral sequences
2. **Scalability**: Can handle larger datasets and models  
3. **Clinical potential**: Spectral features enable tissue segmentation
4. **Production readiness**: System stable for extended training

## Monitoring Progress
Check logs at:
- **TensorBoard**: `outputs/logs_overnight/fold_*/tensorboard/`
- **Metrics**: `outputs/logs_overnight/cv_results.json`
- **Models**: `outputs/models_overnight/fold_*/best_model.pth`

## Next Steps After Completion
1. Evaluate results and generate visualizations
2. Scale to full dataset if promising (remove per-core limits)  
3. Hyperparameter optimization
4. Comparison with other architectures