#!/bin/bash
set -e

echo "============================================"
echo " Phase 1: Environment Preparation"
echo " Gemma 4 Local Inference Stack"
echo "============================================"
echo ""

# 1.1 — Install Homebrew
echo "[1/5] Checking Homebrew..."
if command -v brew &>/dev/null; then
    echo "  Homebrew already installed: $(brew --version | head -1)"
else
    echo "  Installing Homebrew (requires sudo)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to PATH for M-series Macs
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
echo ""

# 1.2 — Install core dependencies
echo "[2/5] Installing core dependencies via Homebrew..."
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install cmake git python@3.12
echo ""

# Verify installations
echo "  Verifying..."
echo "  Homebrew: $(brew --version | head -1)"
echo "  Python:   $(python3.12 --version)"
echo ""

# 1.3 — Python dependencies (MLX-native inference stack + TurboQuant)
echo "[3/5] Installing Python dependencies..."
python3.12 -m pip install huggingface-hub hf-transfer --break-system-packages
python3.12 -m pip install git+https://github.com/ml-explore/mlx-lm --break-system-packages
python3.12 -m pip install mlx-vlm --break-system-packages
python3.12 -m pip install fastapi uvicorn httpx --break-system-packages
echo "  mlx-vlm installed (provides TurboQuant KV cache compression)"
echo ""

# 1.4 — Environment variables
echo "[4/5] Setting environment variables..."

# Only add if not already present
if ! grep -q "HF_HUB_ENABLE_HF_TRANSFER" ~/.zprofile 2>/dev/null; then
    cat >> ~/.zprofile << 'ENVEOF'

# Hugging Face fast transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
ENVEOF
    echo "  Environment variables added to ~/.zprofile"
else
    echo "  Environment variables already configured"
fi
source ~/.zprofile
echo ""

# 1.5 — Pre-download MLX models
echo "[5/5] Pre-downloading E2B and E4B models..."
python3.12 -c "
from huggingface_hub import snapshot_download
print('  Downloading E2B (~2.6 GB)...')
snapshot_download('mlx-community/gemma-4-e2b-it-4bit')
print('  Downloading E4B (~4.3 GB)...')
snapshot_download('mlx-community/gemma-4-e4b-it-4bit')
print('  Done!')
"
echo ""

echo "============================================"
echo " Phase 1 Complete!"
echo " "
echo " Now run: source ~/.zprofile"
echo " Then come back to Claude Code to continue."
echo "============================================"
