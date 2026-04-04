#!/bin/bash
# Start Tier 2 — Gemma 4 E4B (MLX-VLM + TurboQuant, port 8083)
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

MODEL="mlx-community/gemma-4-e4b-it-4bit"
PORT=8083

echo "Starting Primary Tier (E4B) on port $PORT with TurboQuant KV cache..."
python3.12 -m mlx_vlm.server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --kv-bits 3.5 \
    --kv-quant-scheme turboquant &

echo $! > /tmp/mlx_primary.pid
sleep 6
curl -s http://localhost:$PORT/health && echo " Primary tier ready (PID $(cat /tmp/mlx_primary.pid))"
