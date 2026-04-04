#!/bin/bash
# Start Tier 2 — Gemma 4 E4B (MLX, port 8083)
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

MODEL="mlx-community/gemma-4-e4b-it-4bit"
PORT=8083

echo "Starting Primary Tier (E4B) on port $PORT..."
python3.12 -m mlx_lm server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-tokens 1024 \
    --log-level WARNING &

echo $! > /tmp/mlx_primary.pid
sleep 4
curl -s http://localhost:$PORT/health && echo " Primary tier ready (PID $(cat /tmp/mlx_primary.pid))"
