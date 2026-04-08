#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — One-shot environment bootstrap for AI-powered HCA
#
# Usage:
#   bash setup_env.sh              # CUDA 12.1 (default)
#   bash setup_env.sh cpu          # CPU-only build
#   bash setup_env.sh cu118        # CUDA 11.8
# =============================================================================

set -e

CUDA_TAG="${1:-cu121}"          # cu121 | cu118 | cpu
PYTHON_MIN="3.9"
PYTHON_MAX="3.11"
VENV_DIR=".venv"

# ── 0. Check Python version ───────────────────────────────────────────────────
PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 9 ] || [ "$PY_MINOR" -gt 11 ]; then
  echo "❌ Python ${PY_VERSION} detected — this project requires Python 3.9–3.11."
  echo "   Install pyenv or use conda to switch Python version, then re-run."
  exit 1
fi
echo "✅ Python ${PY_VERSION} OK"

# ── 1. Create virtual environment ────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in ${VENV_DIR}/ ..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 2. System-level audio dependency (Linux) ─────────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  if ! dpkg -s portaudio19-dev &>/dev/null 2>&1; then
    echo "Installing portaudio19-dev (requires sudo) ..."
    sudo apt-get install -y portaudio19-dev
  fi
fi

# ── 3. Install PyTorch with the requested compute backend ────────────────────
if [ "$CUDA_TAG" == "cpu" ]; then
  echo "Installing PyTorch (CPU-only) ..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
else
  echo "Installing PyTorch with ${CUDA_TAG} ..."
  pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --quiet
fi

# ── 4. Install project dependencies ──────────────────────────────────────────
echo "Installing project dependencies from requirements.txt ..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!  Activate with:"
echo "   source ${VENV_DIR}/bin/activate"
echo "   python task.py"
