#!/bin/bash
# VibeVoice-ASR Setup Script
# Based on: VibeVoice/docs/vibevoice-asr.md
# This script sets up VibeVoice-ASR environment for speech-to-text processing

set -e

echo "=========================================="
echo "VibeVoice-ASR Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running inside Docker container
is_inside_docker() {
    if [ -f /.dockerenv ] || grep -q 'docker\|lxc' /proc/1/cgroup 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Python version: $python_version"
    else
        print_error "Python3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if Docker is available (only if not inside Docker)
check_docker() {
    if is_inside_docker; then
        print_status "Running inside Docker container"
        return 0
    fi
    
    if command -v docker &> /dev/null; then
        print_status "Docker is available"
        return 0
    else
        print_warning "Docker is not installed"
        return 1
    fi
}

# Check CUDA availability
check_cuda() {
    print_status "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "CUDA is available:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || print_warning "Could not query GPU details"
    else
        print_warning "nvidia-smi not found. GPU may not be available."
        print_warning "For CPU-only inference, the model will run slower."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        apt-get update -qq
        apt-get install -y -qq ffmpeg git
    elif command -v yum &> /dev/null; then
        yum update -y -q
        yum install -y -q ffmpeg git
    elif command -v pacman &> /dev/null; then
        pacman -Sy --noconfirm ffmpeg git
    else
        print_warning "Could not detect package manager. Please install ffmpeg and git manually."
    fi
}

# Setup uv package manager
setup_uv() {
    print_status "Setting up uv package manager..."
    
    if ! command -v uv &> /dev/null; then
        print_status "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    print_status "uv version: $(uv --version)"
}

# Install flash-attention from prebuilt wheels
install_flash_attention() {
    print_status "Installing flash-attention from prebuilt wheels..."
    
    # Detect versions
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "121")
    TORCH_VERSION=$(python -c "import torch; print(f'{torch.__version__.split(\"+\")[0].split(\".\")[0]}.{torch.__version__.split(\"+\")[0].split(\".\")[1]}')" 2>/dev/null || echo "2.5")
    
    print_status "Detected: Python ${PYTHON_VERSION}, CUDA ${CUDA_VERSION}, PyTorch ${TORCH_VERSION}"
    
    # Use prebuilt wheel from GitHub releases
    FLASH_ATTN_VERSION="2.7.4.post1"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_VERSION}torch${TORCH_VERSION}cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
    
    print_status "Downloading prebuilt wheel: ${WHEEL_URL}"
    uv pip install "${WHEEL_URL}" || {
        print_error "Failed to install flash-attention from prebuilt wheel."
        print_error "Please check if a compatible wheel exists at: https://github.com/Dao-AILab/flash-attention/releases"
        print_error "Your system: Python ${PYTHON_VERSION}, CUDA ${CUDA_VERSION}, PyTorch ${TORCH_VERSION}"
        exit 1
    }
    
    print_status "Flash-attention installed successfully from prebuilt wheel"
}

# Create virtual environment and install dependencies
setup_venv() {
    print_status "Creating virtual environment..."
    
    # Check if already in a venv
    if [ -n "$VIRTUAL_ENV" ]; then
        print_status "Already in virtual environment: $VIRTUAL_ENV"
    else
        uv venv --seed
        source .venv/bin/activate
    fi
    
    print_status "Installing VibeVoice-ASR..."
    uv pip install -e .
    
    # Install flash-attention from prebuilt wheels (required, no fallback)
    if ! python -c 'import flash_attn' 2>/dev/null; then
        install_flash_attention
    else
        print_status "Flash attention is already available"
    fi
}

# Print Docker setup instructions
print_docker_instructions() {
    echo ""
    echo "=========================================="
    echo "Docker Setup Instructions (Recommended)"
    echo "=========================================="
    echo ""
    echo "VibeVoice-ASR recommends using NVIDIA Deep Learning Container:"
    echo ""
    echo "1. Launch the container:"
    echo "   sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it nvcr.io/nvidia/pytorch:25.12-py3"
    echo ""
    echo "   Note: Verified versions 24.07 ~ 25.12. Previous versions also compatible."
    echo ""
    echo "2. Inside the container, run this script:"
    echo "   bash setup_vibevoice_asr.sh"
    echo ""
}

# Print usage instructions
print_usage() {
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "VibeVoice-ASR Features:"
    echo "  • 60-minute single-pass audio processing"
    echo "  • Speaker diarization and timestamps"
    echo "  • Customized hotwords support"
    echo "  • 50+ languages with code-switching"
    echo ""
    echo "Usage Examples:"
    echo ""
    echo "1. Launch Gradio Demo:"
    echo "   python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share"
    echo ""
    echo "2. Inference from Audio File:"
    echo "   python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_files <path_to_audio>"
    echo ""
    echo "3. Fine-tuning:"
    echo "   See: finetuning-asr/README.md"
    echo ""
    echo "Documentation:"
    echo "  • Model: https://huggingface.co/microsoft/VibeVoice-ASR"
    echo "  • Demo: https://aka.ms/vibevoice-asr"
    echo "  • Report: https://arxiv.org/pdf/2601.18184"
    echo "  • vLLM: docs/vibevoice-vllm-asr.md"
    echo ""
    
    if ! is_inside_docker; then
        print_docker_instructions
    fi
}

# Main setup function
main() {
    print_status "Starting VibeVoice-ASR setup..."
    
    # Environment checks
    check_python
    check_cuda
    
    # Install system deps
    install_system_deps
    
    # Setup uv and venv
    setup_uv
    setup_venv
    
    # Print final instructions
    print_usage
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "VibeVoice-ASR Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h       Show this help message"
        echo "  --docker         Show Docker setup instructions"
        echo "  --no-venv        Skip virtual environment creation (use existing)"
        echo ""
        echo "This script will:"
        echo "  1. Check Python and CUDA availability"
        echo "  2. Install system dependencies (ffmpeg, git)"
        echo "  3. Setup uv package manager"
        echo "  4. Create virtual environment and install VibeVoice-ASR"
        echo "  5. Install flash-attention (optional, for better performance)"
        echo ""
        echo "For best results, run inside NVIDIA PyTorch Container."
        exit 0
        ;;
    --docker)
        print_docker_instructions
        exit 0
        ;;
    --no-venv)
        print_status "Skipping virtual environment creation..."
        check_python
        check_cuda
        install_system_deps
        setup_uv
        print_status "Installing VibeVoice-ASR in current environment..."
        uv pip install -e .
        print_usage
        exit 0
        ;;
    *)
        main
        ;;
esac
