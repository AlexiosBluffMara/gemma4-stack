#!/bin/bash
# Start Gemma 4 26B-A4B via mlx_vlm server + TurboQuant on MacBook Pro M4 Max
# Serves OpenAI-compatible API on port 8084

eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"
PORT=8084
PID_FILE="/tmp/mlx_heavy.pid"
LOG_FILE="/tmp/mlx_heavy.log"

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null
    rm -f "$PID_FILE"
fi

echo "Starting mlx_vlm server + TurboQuant..."
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "  Host:     0.0.0.0"
echo "  KV bits:  3 (TurboQuant — 4.6x cache compression)"
echo ""

python3.12 -m mlx_vlm.server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --kv-bits 3 \
    --kv-quant-scheme turboquant \
    2>"$LOG_FILE" &

echo $! > "$PID_FILE"

# Wait briefly and confirm it's running
sleep 8
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "mlx_vlm server running — PID $(cat "$PID_FILE")"
    echo ""
    echo "  TurboQuant: 3-bit KV cache (4.6x compression)"
    echo "  Context:    ~256K tokens feasible on 36GB+ M4 Max"
    echo ""
    echo "Monitor: tail -f $LOG_FILE"
    echo "Health:  curl http://localhost:$PORT/health"
    echo "Stop:    bash scripts/stop-macbook.sh"
else
    echo "ERROR: Server failed to start. Check log:"
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
