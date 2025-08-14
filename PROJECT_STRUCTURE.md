# Project Structure (After Cleanup)

## Main Directories

### `/ssmu_net/` - Core Library
- `config.yaml` - Main SSMU-Net configuration
- `config_optimized.yaml` - Configuration with optimized preprocessing
- `preprocess_optimized.py` - Optimized preprocessing pipeline
- `models.py` - SSMU-Net model implementation
- `bench_models.py` - Benchmark model implementations
- `data.py` - Data loading and dataset classes
- `losses.py` - Loss functions
- `evaluation_metrics.py` - Evaluation metrics

### `/scripts/` - Executable Scripts
- `preprocess_data.py` - Main preprocessing script with statistics
- `train_benchmark.py` - Training script for benchmark models
- `visualize_training_data.py` - Data visualization with annotations
- `analyze_preprocessing.py` - Preprocessing comparison analysis
- `cleanup_project.py` - This cleanup script

### `/outputs/` - Generated Data
- `npz/` - Original preprocessed data (15% tissue)
- `npz_optimized/` - Optimized preprocessed data (60% tissue)
- `models/` - Trained model checkpoints
- `figures/` - Generated figures
- `logs/` - Training logs
- `training_visualization/` - Data visualization outputs

### `/archive/` - Archived Files
Contains superseded versions and experimental code for reference.

## Key Improvements Made
1. **Preprocessing**: 4x more training data (60% vs 15% tissue)
2. **Organization**: Clear separation of active vs archived code
3. **Naming**: Removed confusing "fixed", "new", "optimized" suffixes where possible
4. **Documentation**: Clear structure and purpose for each component
