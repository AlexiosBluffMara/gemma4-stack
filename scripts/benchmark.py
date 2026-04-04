"""
Benchmark all active tiers. Run after all servers are up.
"""

import requests
import time
import statistics
import json

TIERS = {
    "fast (E2B)":    ("http://localhost:8082/v1/chat/completions", "mlx-community/gemma-4-e2b-it-4bit"),
    "primary (E4B)": ("http://localhost:8083/v1/chat/completions", "mlx-community/gemma-4-e4b-it-4bit"),
    "heavy (26B)":   ("http://localhost:8081/v1/chat/completions", "mlx-community/gemma-4-26b-a4b-it-4bit"),
}

CLASSIFY_TESTS = [
    "What is the status of the deployment?",
    "Hey, how are you doing today?",
    "Please fix the bug in the login module.",
    "Just FYI the meeting is moved to 3pm.",
    "I had an idea about improving the onboarding flow.",
]

SUMMARIZE_TEST = (
    "The quarterly results showed significant improvement across all key performance "
    "indicators including revenue growth of 23%, customer acquisition up 15%, and "
    "churn reduced to 2.1% from the previous 3.4% last quarter. The engineering team "
    "shipped 47 features and resolved 312 bugs. Infrastructure costs decreased 8% "
    "due to the new caching layer."
)


def call(url, model, content, max_tokens=32):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        t = time.time()
        r = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - t
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text, elapsed
    except Exception as e:
        return f"ERROR: {e}", 999.0


def benchmark_tier(name, url, model):
    print(f"\n{'='*55}")
    print(f"  Tier: {name}")
    print(f"  URL:  {url}")
    print(f"{'='*55}")

    # Warm-up
    call(url, model, "hi", max_tokens=5)

    # Classification speed
    print("\n  Classification (5 messages, max_tokens=10):")
    times = []
    for msg in CLASSIFY_TESTS:
        result, t = call(url, model,
                         f"Classify (question/request/idea/greeting/fyi): {msg}",
                         max_tokens=10)
        times.append(t)
        print(f"    {t:5.2f}s | {result[:18]:18s} | {msg[:45]}")

    print(f"\n  Median: {statistics.median(times):.2f}s  Mean: {statistics.mean(times):.2f}s  "
          f"Min: {min(times):.2f}s  Max: {max(times):.2f}s")

    # Summarization speed
    print(f"\n  Summarization (150-word input, max_tokens=64):")
    result, t = call(url, model,
                     f"Summarize in 20 words: {SUMMARIZE_TEST}",
                     max_tokens=64)
    print(f"    {t:.2f}s | {result[:100]}")


if __name__ == "__main__":
    print("=== Gemma 4 MLX Stack Benchmark ===")
    print("Checking active tiers...")

    active = {}
    for name, (url, model) in TIERS.items():
        base = url.rsplit("/v1", 1)[0]
        try:
            r = requests.get(f"{base}/health", timeout=3)
            if r.status_code == 200:
                active[name] = (url, model)
                print(f"  [OK]  {name}")
            else:
                print(f"  [--]  {name} (unhealthy)")
        except Exception:
            print(f"  [--]  {name} (offline)")

    if not active:
        print("\nNo tiers running. Start with the start_*.sh scripts.")
        exit(1)

    for name, (url, model) in active.items():
        benchmark_tier(name, url, model)

    print("\n\n=== Summary ===")
    print("Target latencies:")
    print("  Fast tier classify:       < 1.5s  (MLX on M4 vs 2.5s plan target)")
    print("  Primary tier summarize:   < 6s    (MLX vs 10s plan target)")
    print("  Heavy tier generation:    > 8 tok/s")
