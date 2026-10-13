#!/bin/sh
# Correctness test for expert streaming, runnable in seconds without a real model.
#
# Generates a tiny random Qwen3-MoE GGUF (tools/make_tiny_moe.py), runs 12 greedy steps fully
# resident, then with streaming under three conditions, and requires byte-identical top logits:
#   stream_small   - a pool far too small to request, so it falls back to the minimum slot count
#                    and evicts constantly
#   stream_big     - every expert fits, so after warm-up everything hits
#   stream_chunked - --n-batch 2 splits the prompt into chunks (multi-step prefill)
# Exit status is non-zero if any variant differs. Needs build/lstk.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP="${TMPDIR:-/tmp}/lstk_test_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
python3 -I "$HERE/tools/make_tiny_moe.py" "$TMP/tiny.gguf" > /dev/null
FILTER='^ggml_metal|^load|^print_info|^llama_model_loader|^init|^llama_|^graph_|^\s*$'
# run ARGS...: fixed raw-token prompt, greedy, printing the top 5 logits after every step.
run() {
    "$HERE/build/lstk" --model "$TMP/tiny.gguf" --tokens 5,17,99,3,42,7 --n-predict 12 --temp 0 --n-ctx 256 --print-logits 5 "$@" 2>&1 | grep -E '^top|^\['
}
run > "$TMP/resident.txt"
run --stream-experts --pool-gib 0.00001 > "$TMP/stream_small.txt"
run --stream-experts --pool-gib 1 > "$TMP/stream_big.txt"
run --n-batch 2 --stream-experts --pool-gib 0.00001 > "$TMP/stream_chunked.txt"
fail=0
for f in stream_small stream_big stream_chunked; do
    if cmp -s "$TMP/resident.txt" "$TMP/$f.txt"; then
        echo "PASS $f: logits identical to resident ($(grep -c '^top' "$TMP/resident.txt") rows)"
    else
        echo "FAIL $f: logits differ from resident"
        diff "$TMP/resident.txt" "$TMP/$f.txt" | head -5
        fail=1
    fi
done
exit $fail
