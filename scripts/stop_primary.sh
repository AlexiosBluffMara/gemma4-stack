#!/bin/bash
if [ -f /tmp/mlx_primary.pid ]; then
    kill $(cat /tmp/mlx_primary.pid) 2>/dev/null && echo "Primary tier (E4B) stopped." || echo "Already stopped."
    rm -f /tmp/mlx_primary.pid
else
    pkill -f "mlx_lm.*8083" 2>/dev/null && echo "Primary tier stopped." || echo "Not running."
fi
