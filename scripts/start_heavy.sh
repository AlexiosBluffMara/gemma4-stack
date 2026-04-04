#!/bin/bash
# Start Tier 3 — Gemma 4 26B-A4B via llama-server + GGUF + mmap
#
# Two modes depending on whether small tiers are stopped:
#   GPU mode (recommended): stop E2B+E4B first → 8-17 tok/s via Metal
#   CPU mode (fallback):    E2B+E4B stay running → ~2 tok/s via BLAS
#
# This script uses GPU mode by default (stops small tiers).
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

GGUF="$HOME/.local/share/llama-models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
PORT=8081
MODE="${1:-gpu}"  # pass "cpu" as $1 to keep small tiers running

if [ ! -f "$GGUF" ]; then
    echo "GGUF not found at $GGUF"
    exit 1
fi

# Kill any existing heavy tier
kill $(cat /tmp/llama_heavy.pid 2>/dev/null) 2>/dev/null
rm -f /tmp/llama_heavy.pid

if [ "$MODE" = "gpu" ]; then
    echo "GPU mode: stopping Fast + Primary tiers to free Metal GPU memory..."
    ~/ai-scripts/stop_fast.sh 2>/dev/null
    ~/ai-scripts/stop_primary.sh 2>/dev/null
    sleep 3
    echo "Starting Heavy Tier (26B-A4B, GPU mode, 5-15 tok/s expected) on port $PORT..."
    echo "NOTE: On 16GB, Metal OOM may occur. If it hangs, use CPU mode instead:"
    echo "  ~/ai-scripts/start_heavy.sh cpu"
    llama-server \
        --model "$GGUF" \
        --port $PORT \
        --host 0.0.0.0 \
        --ctx-size 4096 \
        --mmap \
        --threads 10 \
        --parallel 1 \
        --reasoning off \
        --no-warmup \
        2>/tmp/llama_heavy.log &
else
    echo "CPU mode: E2B+E4B will keep running (expect ~2 tok/s, use for background tasks)"
    llama-server \
        --model "$GGUF" \
        --port $PORT \
        --host 0.0.0.0 \
        --ctx-size 4096 \
        --mmap \
        --threads 10 \
        --gpu-layers 0 \
        --reasoning off \
        2>/tmp/llama_heavy.log &
fi

echo $! > /tmp/llama_heavy.pid
echo "Heavy tier PID: $(cat /tmp/llama_heavy.pid). Loading ~60s..."
echo ""
echo "Monitor: tail -f /tmp/llama_heavy.log"
echo "Ready:   curl http://localhost:$PORT/health"
echo "Stop:    ~/ai-scripts/stop_heavy.sh"
