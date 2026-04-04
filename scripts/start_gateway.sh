#!/bin/bash
# Start the FastAPI gateway on port 8080 (all interfaces — accessible over Tailscale)
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile 2>/dev/null

# Install gateway deps if needed
python3.12 -m pip show fastapi uvicorn httpx &>/dev/null || \
    python3.12 -m pip install fastapi uvicorn httpx --break-system-packages

echo "Starting API gateway on port 8080..."
cd ~/ai-scripts
python3.12 -m uvicorn gateway:app --host 0.0.0.0 --port 8080 --log-level warning &
echo $! > /tmp/mlx_gateway.pid
sleep 2
curl -s http://localhost:8080/health | python3.12 -m json.tool
echo "Gateway running (PID $(cat /tmp/mlx_gateway.pid))"
echo "Access via Tailscale: http://<tailscale-ip>:8080"
