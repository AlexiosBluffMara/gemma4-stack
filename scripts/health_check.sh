#!/bin/bash
eval "$(/opt/homebrew/bin/brew shellenv)"

echo "=== Gemma 4 Stack Health Check ==="
echo ""

check_tier() {
    local name=$1
    local port=$2
    local result
    result=$(curl -s --max-time 3 "http://localhost:$port/health" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$result" ]; then
        echo "[OK]   $name on port $port"
    else
        echo "[OFF]  $name on port $port — start with ~/ai-scripts/start_$(echo $name | tr '[:upper:]' '[:lower:]' | cut -d' ' -f1).sh"
    fi
}

check_tier "Fast tier (E2B)"    8082
check_tier "Primary tier (E4B)" 8083
check_tier "Heavy tier (26B)"   8081
check_tier "Gateway"            8080

echo ""
echo "=== Memory ==="
vm_stat | awk '
  /Pages free/    {free=$3}
  /Pages active/  {active=$3}
  /Pages wired/   {wired=$4}
  /Pages occupied by compressor/ {comp=$5}
  END {
    page=4096
    printf "  Free:     %.1f GB\n", free*page/1073741824
    printf "  Active:   %.1f GB\n", active*page/1073741824
    printf "  Wired:    %.1f GB\n", wired*page/1073741824
    printf "  Compressed: %.1f GB\n", comp*page/1073741824
  }'

echo ""
echo "=== Running MLX Processes ==="
pgrep -la python3 | grep mlx || echo "  None"

echo ""
echo "=== Disk ==="
df -h / | tail -1 | awk '{printf "  Used: %s / %s  (%s full)\n", $3, $2, $5}'
du -sh ~/.cache/huggingface/hub/ 2>/dev/null | awk '{print "  HF cache: "$1}'
