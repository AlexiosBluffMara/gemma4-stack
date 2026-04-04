#!/bin/bash
# Stop the mlx_lm server on MacBook Pro

PID_FILE="/tmp/mlx_heavy.pid"

if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null && echo "mlx_lm server stopped." || echo "Already stopped."
    rm -f "$PID_FILE"
else
    pkill -f "mlx_lm server.*8084" 2>/dev/null && echo "mlx_lm server stopped." || echo "Not running."
fi
