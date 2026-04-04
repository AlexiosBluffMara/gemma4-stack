#!/bin/bash
set -e
source ~/.zprofile 2>/dev/null

echo "============================================"
echo " Phase 3: Download Gemma 4 26B-A4B (Heavy)"
echo "============================================"
echo ""

mkdir -p ~/.local/share/llama-models
cd ~/.local/share/llama-models

echo "[1/3] Downloading Gemma 4 26B-A4B GGUF from Unsloth..."
echo "  This is ~16-18 GB. Expected: 20-45 minutes."
echo "  Downloads auto-resume if interrupted."
echo ""

python3.12 -c "
from huggingface_hub import hf_hub_download
import os

target = os.path.expanduser('~/.local/share/llama-models')

print('Downloading main model weights...')
hf_hub_download(
    repo_id='unsloth/gemma-4-26B-A4B-it-GGUF',
    filename='gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf',
    local_dir=target
)

print('Downloading vision projector...')
hf_hub_download(
    repo_id='unsloth/gemma-4-26B-A4B-it-GGUF',
    filename='mmproj-BF16.gguf',
    local_dir=target
)
print('Download complete!')
"

echo ""
echo "[2/3] Verifying files..."
ls -lh ~/.local/share/llama-models/
echo ""

echo "[3/3] Disk status:"
df -h / | tail -1
echo ""

echo "============================================"
echo " Phase 3 Complete!"
echo " Start heavy tier with: ~/ai-scripts/start_heavy.sh"
echo "============================================"
