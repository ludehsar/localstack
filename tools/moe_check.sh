#!/bin/sh
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
M="$1"
OUT="${2:-$HERE/docs/runs}"
mkdir -p "$OUT"
NAME=$(basename "$M" .gguf)
LOG="$OUT/$NAME.md"
TMP="${TMPDIR:-/tmp}/moe_check_$$"
mkdir -p "$TMP"
FILTER='^ggml_metal|^load|^print_info|^llama_model_loader|^init|^llama_|^graph_|^\s*$'
AVAIL=$(vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} END{gsub(/\./,"",f); gsub(/\./,"",i); printf "%.1f", (f+i)*16384/1073741824}')
PROMPT="Explain in detail how a hash map handles collisions, and compare chaining with open addressing."
run() { "$HERE/build/lstk" --model "$M" --n-ctx 2048 "$@" 2>&1 | grep -vE "$FILTER"; }

{
    echo "# $NAME"
    echo
    echo "Run: $(date '+%Y-%m-%d %H:%M'), available memory (free+inactive) at start: ${AVAIL} GiB"
    echo
    echo "## Streaming self-consistency (greedy, pool 1 GiB vs 2.5 GiB)"
    run --prompt "$PROMPT" --n-predict 24 --temp 0 --print-logits 3 --stream-experts --pool-gib 1 > "$TMP/a.txt"
    run --prompt "$PROMPT" --n-predict 24 --temp 0 --print-logits 3 --stream-experts --pool-gib 2.5 > "$TMP/b.txt"
    echo '```'
    grep -E "^model|^experts|^memory|^prompt:" "$TMP/a.txt"
    echo "--- pool 2.5 GiB ---"
    grep -E "^experts|^prompt:" "$TMP/b.txt"
    echo '```'
    if diff <(grep -E "^top" "$TMP/a.txt") <(grep -E "^top" "$TMP/b.txt") > /dev/null; then
        echo "Result: top logits identical across pool sizes ($(grep -c '^top' "$TMP/a.txt") rows): PASS"
    else
        echo "Result: top logits DIFFER across pool sizes: FAIL"
    fi
    echo
    echo "Generated text (pool 1 GiB):"
    echo '```'
    grep -vE "^top|^model|^experts|^memory|^prompt:|^  pool" "$TMP/a.txt" | head -5
    echo '```'
    echo
    echo "## Routing trace (200 sampled tokens) and cache simulation"
    run --prompt "$PROMPT" --n-predict 200 --temp 0.8 --seed 1 --stream-experts --pool-gib 2 --trace-routing "$TMP/trace.bin" > "$TMP/trace_run.txt"
    echo '```'
    grep -E "^experts: [0-9]|^prompt:" "$TMP/trace_run.txt"
    python3 -I "$HERE/tools/cache_sim.py" "$TMP/trace.bin"
    echo '```'
    cp "$TMP/trace.bin" "$OUT/$NAME.trace.bin"
    echo
    echo "## Decode speed by pool size (--bench 64 48, streaming)"
    echo '```'
    for POOL in 1 2 3 4; do
        echo "pool ${POOL} GiB:"
        run --bench 64 48 --stream-experts --pool-gib "$POOL" | grep -E "^bench|^experts: [0-9]|^memory pressure"
    done
    echo '```'
    if [ "$(echo "$AVAIL >= 13" | bc)" = "1" ]; then
        echo
        echo "## Resident (all experts in RAM) and llama.cpp baseline"
        echo '```'
        run --bench 64 48 | grep -E "^bench|^memory"
        "$HERE/third_party/llama.cpp/build/bin/llama-bench" -m "$M" -p 64 -n 48 -fa 1 -r 1 2>/dev/null | tail -3
        echo '```'
    else
        echo
        echo "Resident run and llama.cpp baseline skipped: only ${AVAIL} GiB available (need about 13 GiB free for this file)."
    fi
} > "$LOG" 2>&1
rm -rf "$TMP"
cat "$LOG"
