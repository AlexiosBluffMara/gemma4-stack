#!/bin/bash
set -e

echo "============================================"
echo " MacBook Pro M4 Max — Gemma 4 Node Setup"
echo " Joins the inference mesh as MLX heavy tier"
echo "============================================"
echo ""

MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"

# ── 1. Verify macOS + Apple Silicon ──────────────────────────────
echo "[1/5] Checking platform..."
if [ "$(uname)" != "Darwin" ]; then
    echo "  ERROR: This script requires macOS."
    exit 1
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "  ERROR: Apple Silicon required (detected: $ARCH)."
    exit 1
fi
echo "  macOS $(sw_vers -productVersion) on $ARCH — OK"
echo ""

# ── 2. Install Homebrew ──────────────────────────────────────────
echo "[2/5] Checking Homebrew..."
if command -v brew &>/dev/null; then
    echo "  Homebrew already installed: $(brew --version | head -1)"
else
    echo "  Installing Homebrew (requires sudo)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
echo ""

# ── 3. Install Python 3.12 ──────────────────────────────────────
echo "[3/5] Installing Python 3.12..."
brew install python@3.12
echo "  Python: $(python3.12 --version)"
echo ""

# ── 4. Install MLX packages ─────────────────────────────────────
echo "[4/5] Installing mlx-lm (from main) and mlx-vlm..."
python3.12 -m pip install git+https://github.com/ml-explore/mlx-lm --break-system-packages
python3.12 -m pip install mlx-vlm --break-system-packages
python3.12 -m pip install huggingface-hub hf-transfer --break-system-packages
echo ""

# ── 5. Pre-download model ───────────────────────────────────────
echo "[5/5] Downloading model: $MODEL (~15.6 GB)..."
echo "  This may take a while on first run."
export HF_HUB_ENABLE_HF_TRANSFER=1
python3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
print('  Download complete.')
"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "============================================"
echo " Setup Complete!"
echo ""
echo " Model: $MODEL"
echo "        Downloaded and ready."
echo ""
echo " Start the server:"
echo "   bash scripts/start-macbook.sh"
echo ""
echo " Reminder: install and configure Tailscale"
echo "   so the Mac Mini gateway can reach this"
echo "   node over the mesh network."
echo "   brew install tailscale"
echo "   See: https://tailscale.com/download/mac"
echo "============================================"
