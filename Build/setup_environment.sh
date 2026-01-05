#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh
# Environment Setup Script for the Deep learning Hybrid Model framework for research on Camouflage and Concealment Technology based on Deeplearning
#
# This script:
#   - Creates a fresh conda environment
#   - Installs PyTorch with CUDA 12.1 support
#   - Installs project-specific dependencies with pinned versions
#   - Verifies key imports (PyTorch, CUDA, YOLOv8)
#
# Usage:
#   bash setup_environment.sh
#   # or
#   ./setup_environment.sh
#
# Notes:
#   - Requires conda to be installed and available in PATH
#   - Designed for Linux/macOS (Windows users should use WSL or adapt)
#   - Pinned versions chosen for known compatibility (as of late 2024/early 2025)
# =============================================================================

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# ─── Configuration ────────────────────────────────────────────────────────────

ENV_NAME="invisibility_cloak"
PYTHON_VERSION="3.11"

# PyTorch + CUDA version (adjust index-url if you need different CUDA)
PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
PYTORCH_VERSION="2.1.0"
TORCHVISION_VERSION="0.16.0"

# ─── Helper Functions ─────────────────────────────────────────────────────────

print_header() {
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│ $1"
    echo "└──────────────────────────────────────────────────────────────┘"
}

print_step() {
    echo "→ $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not found." >&2
        exit 1
    fi
}

# ─── Main Execution ───────────────────────────────────────────────────────────

print_header "Invisibility Cloaking Environment Setup"

# Check prerequisites
check_command conda
check_command python

print_step "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

print_step "Activating environment"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" || true
conda activate "$ENV_NAME" || { echo "Failed to activate environment"; exit 1; }

print_step "Installing PyTorch ${PYTORCH_VERSION} + torchvision with CUDA ${PYTORCH_INDEX##*/}"
pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --index-url "${PYTORCH_INDEX}"

print_step "Installing core dependencies"
pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    ultralytics==8.0.200 \
    scikit-image==0.21.0 \
    pillow==10.0.1 \
    tqdm==4.66.1 \
    matplotlib==3.8.0

print_step "Installing deep learning / augmentation libraries"
pip install --no-cache-dir \
    timm==0.9.7 \
    segmentation-models-pytorch==0.3.3 \
    albumentations==1.3.1

print_step "Installing LaMa / kornia dependencies"
pip install --no-cache-dir \
    easydict==1.10 \
    kornia==0.7.0

print_header "Verifying Installation"

echo "PyTorch version:"
python -c "import torch; print('  PyTorch:', torch.__version__)"

echo "CUDA availability:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('  CUDA device count:', torch.cuda.device_count())" 2>/dev/null || echo "  (no CUDA devices detected)"

echo "YOLOv8 import check:"
python -c "from ultralytics import YOLO; print('  YOLOv8: OK')"

print_header "Setup Complete!"
echo ""
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. git clone your repo (if not already done)"
echo "  3. python your_main_script.py"
echo ""
echo "To deactivate later: conda deactivate"
echo "To remove environment: conda env remove -n ${ENV_NAME}"
