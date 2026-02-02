#!/bin/bash
set -e

echo "Setting up VibeVoice environment with uv..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Check and install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Create virtual environment with uv
echo "Creating virtual environment..."
uv venv --seed

# Install PyTorch (CUDA 12.1 - adjust as needed)
echo "Installing PyTorch..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention from prebuilt wheels
echo "Installing flash-attention from prebuilt wheels..."
uv pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases

# Install other dependencies from pyproject.toml
echo "Installing VibeVoice and dependencies..."
uv pip install -e .

# Install vLLM (optional - for ASR inference)
echo "Installing vLLM..."
uv pip install vllm

echo ""
echo "Setup complete! VibeVoice is ready to use."
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use uv run directly:"
echo "  uv run python demo/vibevoice_realtime_demo.py"
echo ""
echo "To sync dependencies later:"
echo "  uv pip install -e ."
