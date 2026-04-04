"""
Three-tier Gemma 4 local inference router.

Tier 1 (Fast):    Gemma 4 E2B  via mlx_lm server  :8082 — triage, classification
                  MLX native Metal GPU · 126 tok/s · 2.6 GB RAM
Tier 2 (Primary): Gemma 4 E4B  via mlx_lm server  :8083 — summarization, compression
                  MLX native Metal GPU · 32 tok/s  · 4.3 GB RAM
Tier 3 (Heavy):   Gemma 4 26B  via llama-server    :8081 — complex tasks, on-demand
                  llama.cpp + GGUF + mmap · ~8-17 tok/s · only ~4-5 GB RAM active
                  (mmap pages rest of 16-18 GB model from SSD as needed)

Tiers 1+2 run always. Tier 3 is on-demand (start_heavy.sh).
All expose OpenAI-compatible /v1/chat/completions.
"""

import requests
import json
import time

# Tier endpoints
TIER_FAST    = "http://localhost:8082/v1/chat/completions"
TIER_PRIMARY = "http://localhost:8083/v1/chat/completions"
TIER_HEAVY   = "http://localhost:8081/v1/chat/completions"

# Model names (used in API calls)
MODEL_FAST    = "mlx-community/gemma-4-e2b-it-4bit"
MODEL_PRIMARY = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_HEAVY   = "gemma-4-26B"  # served by llama-server, model name is arbitrary


def _call(url, model, messages, max_tokens=256, timeout=60):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def classify(text):
    """Tier 1 — classify message type in <2s."""
    return _call(
        TIER_FAST, MODEL_FAST,
        [{"role": "user", "content":
          f"Classify into exactly one: question, request, idea, greeting, fyi\n"
          f"Message: {text}\nReply with only the category word."}],
        max_tokens=10,
        timeout=15,
    )


def compress(text, words=30):
    """Tier 2 — compress long text to ~N words."""
    return _call(
        TIER_PRIMARY, MODEL_PRIMARY,
        [{"role": "user", "content":
          f"Compress to {words} words, preserve key meaning:\n\n{text}"}],
        max_tokens=128,
        timeout=30,
    )


def heavy_query(prompt, max_tokens=1024):
    """Tier 3 — full reasoning on complex tasks."""
    return _call(
        TIER_HEAVY, MODEL_HEAVY,
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=180,
    )


def route(text):
    """
    Auto-route: classify first, then send to appropriate tier.
    Returns (tier_used, response).
    """
    category = classify(text)

    if category in ("greeting", "fyi"):
        # Simple — fast tier handles it
        response = _call(
            TIER_FAST, MODEL_FAST,
            [{"role": "user", "content": text}],
            max_tokens=256,
            timeout=30,
        )
        return "fast", response

    elif category in ("question", "request"):
        # Try primary tier first
        response = _call(
            TIER_PRIMARY, MODEL_PRIMARY,
            [{"role": "user", "content": text}],
            max_tokens=512,
            timeout=60,
        )
        return "primary", response

    else:
        # idea or unknown — route to heavy
        response = heavy_query(text)
        return "heavy", response


if __name__ == "__main__":
    print("=== Testing Fast Tier (classify) ===")
    t = time.time()
    result = classify("What is the status of the deployment?")
    print(f"  Result: {result!r}  ({time.time()-t:.1f}s)")

    print("\n=== Testing Primary Tier (compress) ===")
    t = time.time()
    result = compress(
        "The quarterly results showed significant improvement across all key "
        "performance indicators including revenue growth of 23%, customer "
        "acquisition up 15%, and churn reduced to 2.1% from 3.4% last quarter."
    )
    print(f"  Result: {result!r}  ({time.time()-t:.1f}s)")

    print("\n=== Testing Route ===")
    t = time.time()
    tier, response = route("Hey, how are you doing today?")
    print(f"  Tier: {tier}")
    print(f"  Response: {response[:120]!r}  ({time.time()-t:.1f}s)")
