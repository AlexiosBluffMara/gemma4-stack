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
brew install ollama llama.cpp cmake git python@3.12
echo ""

# Verify installations
echo "  Verifying..."
echo "  Homebrew: $(brew --version | head -1)"
echo "  Ollama:   $(ollama --version 2>&1)"
echo "  llama.cpp: $(llama-server --version 2>&1 | head -1)"
echo "  Python:   $(python3.12 --version)"
echo ""

# 1.3 — Python dependencies
echo "[3/5] Installing Python dependencies..."
python3.12 -m pip install huggingface-hub hf-transfer --break-system-packages
echo ""

# 1.4 — Environment variables
echo "[4/5] Setting environment variables..."

# Only add if not already present
if ! grep -q "HF_HUB_ENABLE_HF_TRANSFER" ~/.zprofile 2>/dev/null; then
    cat >> ~/.zprofile << 'ENVEOF'

# Hugging Face fast transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Ollama inference settings for 16GB Mac Mini
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_MAX_LOADED_MODELS=1
ENVEOF
    echo "  Environment variables added to ~/.zprofile"
else
    echo "  Environment variables already configured"
fi
source ~/.zprofile
echo ""

# 1.5 — Create model storage
echo "[5/5] Creating directories..."
mkdir -p ~/.local/share/llama-models
mkdir -p ~/ai-scripts
echo "  ~/.local/share/llama-models created"
echo "  ~/ai-scripts created"
echo ""

echo "============================================"
echo " Phase 1 Complete!"
echo " "
echo " Now run: source ~/.zprofile"
echo " Then come back to Claude Code to continue."
echo "============================================"
