#!/bin/sh
# Download the GGUF models used by the plan into $MODELS_DIR (default ~/Models/gguf), outside the repo.
#
# Built for a slow, flaky connection and multi-GB files: skips files whose size already matches the
# server's Content-Length, resumes partial files (curl -C -), aborts transfers that stall below
# 10 KB/s for 60 s, and retries up to 200 times. Safe to re-run at any time.
#
#   ./tools/fetch_models.sh            # or: MODELS_DIR=/Volumes/ext/gguf ./tools/fetch_models.sh
#   ./tools/fetch_models.sh Coder      # only files whose name contains "Coder"
set -u
DEST="${MODELS_DIR:-$HOME/Models/gguf}"
FILTER="${1:-}"
mkdir -p "$DEST"
# fetch NAME URL: download URL to $DEST/NAME, resuming and retrying until complete.
fetch() {
    name="$1"
    url="$2"
    case "$name" in *"$FILTER"*) ;; *) return 0 ;; esac
    expected=$(curl -sIL --http1.1 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
    have=$(stat -f %z "$DEST/$name" 2>/dev/null || echo 0)
    if [ -n "$expected" ] && [ "$have" = "$expected" ]; then
        echo "DONE $name $have bytes (already complete)"
        return 0
    fi
    attempt=0
    while [ "$attempt" -lt 200 ]; do
        attempt=$((attempt + 1))
        if curl -sS -L --http1.1 -C - --connect-timeout 30 --speed-time 60 --speed-limit 10000 -o "$DEST/$name" "$url"; then
            echo "DONE $name $(stat -f %z "$DEST/$name") bytes"
            return 0
        fi
        echo "retry $attempt $name at $(stat -f %z "$DEST/$name" 2>/dev/null || echo 0) bytes"
        sleep 3
    done
    echo "FAILED $name"
    return 1
}
# Dense model for parity checks against llama.cpp; then Qwen3-30B-A3B MoE at 2-bit (11.8 GB) and 4-bit (17.7 GB).
fetch Qwen3-1.7B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
fetch Qwen3-30B-A3B-UD-Q2_K_XL.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-Q2_K_XL.gguf
fetch Qwen3-30B-A3B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
# Coding target (same qwen3moe shape as Qwen3-30B-A3B), 4-bit, 17.7 GB.
fetch Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf
ls -la "$DEST"
