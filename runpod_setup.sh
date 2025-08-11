#!/bin/bash
# RunPod setup script for SSMU-Net

echo "🚀 Setting up SSMU-Net on RunPod..."

# Check CUDA version
echo "📍 Checking CUDA version..."
nvcc --version

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip

# Install Mamba SSM first (critical dependency)
echo "🔧 Installing Mamba SSM..."
pip install mamba-ssm

# Install other requirements
echo "📚 Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mamba_ssm; print('Mamba SSM: OK')"

# Check data
echo "📊 Checking NPZ data..."
NPZ_COUNT=$(ls outputs/npz/*.npz 2>/dev/null | wc -l)
echo "Found $NPZ_COUNT NPZ files"

if [ "$NPZ_COUNT" -eq 143 ]; then
    echo "✅ All data files present"
else
    echo "⚠️  Expected 143 NPZ files, found $NPZ_COUNT"
fi

# Check GPU
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "✨ Setup complete! You can now run:"
echo "  python scripts/run_train.py"
echo "  or"
echo "  make train"