#!/bin/bash
# Start Tier 1 — Gemma 4 E2B (MLX-VLM + TurboQuant, port 8082)
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

MODEL="mlx-community/gemma-4-e2b-it-4bit"
PORT=8082

echo "Starting Fast Tier (E2B) on port $PORT with TurboQuant KV cache..."
python3.12 -m mlx_vlm.server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --kv-bits 4 \
    --kv-quant-scheme turboquant &

echo $! > /tmp/mlx_fast.pid
sleep 6
curl -s http://localhost:$PORT/health && echo " Fast tier ready (PID $(cat /tmp/mlx_fast.pid))"
