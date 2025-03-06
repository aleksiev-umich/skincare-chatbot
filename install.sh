#!/bin/bash
python3.11 -m pip install setuptools wheel
python3.11 -m pip install numpy==1.24.3
echo "Installing PyTorch 2.2.0..."
python3.11 -m pip install torch==2.2.0 torchvision>=0.17.0 torchaudio==2.2.0

echo "Installing other requirements..."
python3.11 -m pip install transformers datasets gradio sentence-transformers pandas scikit-learn

echo "Verifying PyTorch installation..."
python3.11 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Installation complete!"