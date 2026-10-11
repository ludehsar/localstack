#!/bin/sh
set -u
DEST="${MODELS_DIR:-$HOME/Models/gguf}"
mkdir -p "$DEST"
fetch() {
    name="$1"
    url="$2"
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
fetch Qwen3-1.7B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
fetch Qwen3-30B-A3B-UD-Q2_K_XL.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-Q2_K_XL.gguf
fetch Qwen3-30B-A3B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
ls -la "$DEST"
