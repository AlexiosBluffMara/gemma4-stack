#!/bin/bash
if [ -f /tmp/llama_heavy.pid ]; then
    kill $(cat /tmp/llama_heavy.pid) 2>/dev/null && echo "Heavy tier (26B llama-server) stopped." || echo "Already stopped."
    rm -f /tmp/llama_heavy.pid
else
    pkill -f "llama-server.*$PORT" 2>/dev/null
    pkill -f "llama-server.*8081" 2>/dev/null && echo "Heavy tier stopped." || echo "Not running."
fi
