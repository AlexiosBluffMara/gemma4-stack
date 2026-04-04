#!/usr/bin/env python3
"""
Download the Gemma 4 26B-A4B GGUF from Unsloth (Hugging Face).
Run this before starting the heavy tier or Docker compose.

Usage:
    python3 download_heavy.py [--dir /path/to/models]

The file lands at:
    ./models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf  (~16 GB)
    ./models/mmproj-BF16.gguf                       (~1.1 GB)
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download Gemma 4 26B-A4B GGUF")
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/.local/share/llama-models"),
        help="Directory to download models into (default: ~/.local/share/llama-models)"
    )
    parser.add_argument(
        "--skip-mmproj",
        action="store_true",
        help="Skip downloading the vision projector (text-only inference)"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface-hub not installed. Run: pip install huggingface-hub hf-transfer")
        sys.exit(1)

    os.makedirs(args.dir, exist_ok=True)
    print(f"Downloading to: {args.dir}")

    print("\n[1/2] Downloading 26B-A4B GGUF (Q4_K_XL, ~16 GB)...")
    print("      This will take 20-45 minutes. Downloads auto-resume if interrupted.")
    hf_hub_download(
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        local_dir=args.dir,
    )
    print("      ✓ Main model downloaded")

    if not args.skip_mmproj:
        print("\n[2/2] Downloading vision projector (mmproj, ~1.1 GB)...")
        hf_hub_download(
            repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
            filename="mmproj-BF16.gguf",
            local_dir=args.dir,
        )
        print("      ✓ Vision projector downloaded")

    import subprocess
    result = subprocess.run(["du", "-sh", args.dir], capture_output=True, text=True)
    print(f"\nTotal size: {result.stdout.split()[0]}")
    print(f"\nModel path for Docker: mount {args.dir} → /models")
    print(f"Model path for native: {args.dir}/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")

if __name__ == "__main__":
    main()
