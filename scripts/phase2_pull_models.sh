#!/bin/bash
set -e
source ~/.zprofile 2>/dev/null

echo "============================================"
echo " Phase 2: Pull Gemma 4 E2B & E4B via Ollama"
echo "============================================"
echo ""

# Ensure Ollama is running
echo "[1/4] Starting Ollama service..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    ollama serve &
    sleep 5
fi
echo "  Ollama running on port 11434"
echo ""

# Pull E2B (Fast Tier)
echo "[2/4] Pulling Gemma 4 E2B (Fast Tier, ~7.2 GB)..."
echo "  This will take 8-15 minutes..."
ollama pull gemma4:e2b
echo "  E2B pulled successfully."
echo ""

# Pull E4B (Primary Tier)
echo "[3/4] Pulling Gemma 4 E4B (Primary Tier, ~9.6 GB)..."
echo "  This will take 10-20 minutes..."
ollama pull gemma4:e4b
echo "  E4B pulled successfully."
echo ""

# Verify
echo "[4/4] Verifying models..."
ollama list
echo ""

echo "Disk usage:"
du -sh ~/.ollama/models/ 2>/dev/null || echo "  (models directory not found at expected path)"
df -h / | tail -1
echo ""

echo "Quick smoke test — E2B classify..."
RESULT=$(ollama run gemma4:e2b "What is 2+2? Reply in one word." --nowordwrap 2>/dev/null)
echo "  E2B response: $RESULT"

echo ""
echo "Quick smoke test — E4B summarize..."
RESULT=$(ollama run gemma4:e4b "Summarize in 10 words: The quick brown fox jumped over the lazy dog." --nowordwrap 2>/dev/null)
echo "  E4B response: $RESULT"

echo ""
echo "============================================"
echo " Phase 2 Complete!"
echo "============================================"
