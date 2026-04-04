#!/bin/bash
# Tailscale setup for Gemma 4 mesh access
# NOTE: The Homebrew formula is CLI-only. macOS requires the official pkg
# for the Network Extension (daemon). Run this script AFTER installing:
#   https://pkgs.tailscale.com/stable/#macos
eval "$(/opt/homebrew/bin/brew shellenv)"

echo "=== Tailscale Setup for Gemma 4 Gateway ==="
echo ""

# Check if Tailscale app is installed
if ! tailscale status &>/dev/null; then
    echo "Tailscale daemon not running."
    echo ""
    echo "macOS requires the official Tailscale pkg installer (not Homebrew)."
    echo "The Network Extension must be approved in System Settings."
    echo ""
    echo "Steps:"
    echo "  1. Download: https://pkgs.tailscale.com/stable/#macos"
    echo "  2. Open the .pkg and install"
    echo "  3. Open System Settings > Privacy & Security"
    echo "  4. Approve the Tailscale Network Extension"
    echo "  5. Re-run this script"
    exit 1
fi

# Connect to Tailscale
echo "[1/3] Authenticating with Tailscale..."
tailscale up --accept-routes

# Get our Tailscale IP
TS_IP=$(tailscale ip -4 2>/dev/null)
echo ""
echo "[2/3] Connected! Your Tailscale IP: $TS_IP"

# Write the IP to a config file for the gateway to use
echo "$TS_IP" > /tmp/tailscale_ip

echo ""
echo "[3/3] Access points on your Tailscale network:"
echo ""
echo "  OpenAI-compatible gateway:  http://$TS_IP:8080/v1/chat/completions"
echo "  Auto-routing gateway:       http://$TS_IP:8080"
echo "  Health check:               http://$TS_IP:8080/health"
echo "  Metrics:                    http://$TS_IP:8080/metrics"
echo "  Fast tier direct (E2B):     http://$TS_IP:8082"
echo "  Primary tier direct (E4B):  http://$TS_IP:8083"
echo "  Heavy tier direct (26B):    http://$TS_IP:8081 (start manually)"
echo ""
echo "=== Quick test from any device on your Tailscale network ==="
echo ""
cat << EXAMPLE
curl http://$TS_IP:8080/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 200
  }'
EXAMPLE
echo ""
echo "=== Use with any OpenAI-compatible client or tool ==="
echo "  Base URL: http://$TS_IP:8080"
echo "  API Key:  not required"
echo "  Model:    leave empty or use any string (auto-routed)"
