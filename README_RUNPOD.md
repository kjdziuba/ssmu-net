# SSMU-Net for RunPod

Hyperspectral tissue segmentation using Sinc filters + Mamba SSM + U-Net.

## Quick Start on RunPod

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ssmu-net-runpod.git
cd ssmu-net-runpod
```

### 2. Install Dependencies
```bash
# Install Mamba SSM (requires CUDA 11.8 or 12.1)
pip install mamba-ssm

# Install other requirements
pip install -r requirements.txt
```

### 3. Verify Data
```bash
# Check that NPZ files are present
ls outputs/npz/*.npz | wc -l  # Should show 143 files
```

### 4. Run Training
```bash
# Full 5-fold cross-validation
python scripts/run_train.py

# Or use Makefile
make train
```

### 5. Monitor Progress
```bash
# Check training logs
tail -f outputs/logs/fold_0/training.log

# Or use tensorboard
tensorboard --logdir outputs/logs
```

## Included Data

This repository includes 143 preprocessed NPZ files in `outputs/npz/`:
- Each file is a tissue core (e.g., `core_C2.npz`)
- Ferguson Order-A normalized
- 451 spectral channels (900-1800 cm⁻¹)
- 8 tissue classes + background
- Z-score statistics in `outputs/tables/zscore_stats.csv`

## RunPod GPU Requirements

- **Minimum**: RTX 3090 (24GB VRAM)
- **Recommended**: A100 (40GB) or H100
- **CUDA**: 11.8 or 12.1 (for Mamba SSM)

## Expected Training Time

- **RTX 3090**: ~4-5 hours for 5-fold CV
- **A100**: ~2-3 hours for 5-fold CV
- **H100**: ~1-2 hours for 5-fold CV

## Configuration

Edit `ssmu_net/config.yaml` to adjust:
- `optim.batch_size`: Reduce if OOM (default: 1)
- `optim.epochs`: Default 120
- `optim.lr`: Default 3e-4

## Outputs

After training completes:
```
outputs/
├── models/
│   └── fold_*/checkpoint_best.pth  # Best models
├── tables/
│   └── cross_validation_results.csv  # Performance metrics
└── logs/
    └── fold_*/training_history.csv  # Training curves
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# Edit ssmu_net/config.yaml
optim:
  batch_size: 1  # Already set to 1, can't go lower
  grad_accumulation: 4  # Add this to simulate larger batches
```

### Mamba SSM Import Error
```bash
# Check CUDA version
nvcc --version

# Reinstall for specific CUDA
pip uninstall mamba-ssm
pip install mamba-ssm --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps After Training

1. **Evaluate best model**:
   ```bash
   python scripts/run_eval.py --checkpoint outputs/models/fold_0/checkpoint_best.pth
   ```

2. **Export DFIR bands**:
   ```bash
   python scripts/run_export_dfir.py
   ```

3. **Generate figures**:
   ```bash
   python scripts/run_figures.py
   ```