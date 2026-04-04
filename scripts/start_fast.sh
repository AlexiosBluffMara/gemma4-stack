#!/bin/bash
# Start Tier 1 — Gemma 4 E2B (MLX, port 8082)
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

MODEL="mlx-community/gemma-4-e2b-it-4bit"
PORT=8082

echo "Starting Fast Tier (E2B) on port $PORT..."
python3.12 -m mlx_lm server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-tokens 512 \
    --log-level WARNING &

echo $! > /tmp/mlx_fast.pid
sleep 4
curl -s http://localhost:$PORT/health && echo " Fast tier ready (PID $(cat /tmp/mlx_fast.pid))"
