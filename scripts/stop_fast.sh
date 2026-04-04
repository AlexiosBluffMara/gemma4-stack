#!/bin/bash
if [ -f /tmp/mlx_fast.pid ]; then
    kill $(cat /tmp/mlx_fast.pid) 2>/dev/null && echo "Fast tier (E2B) stopped." || echo "Already stopped."
    rm -f /tmp/mlx_fast.pid
else
    pkill -f "mlx_lm.*8082" 2>/dev/null && echo "Fast tier stopped." || echo "Not running."
fi
