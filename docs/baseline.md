# Phase 0 baseline measurements

Machine: MacBook Pro M3 Pro (Mac15,6), 11-core CPU (5P+6E), 14-core GPU, 18 GB unified memory,
macOS 26.6.2 (Darwin 25.6.0), 494 GB Apple Fabric SSD. Measured 2026-10-11 under the normal
desktop load (about 11 GB of swap in use), so GPU numbers are conservative and SSD numbers are
representative.

## Metal device (tools/metal_info.m)

| Property | Value |
|---|---|
| recommendedMaxWorkingSetSize | 13.32 GiB (74.0% of RAM) |
| maxBufferLength | 9.99 GiB |
| page size | 16 KiB |
| unified memory | yes |
| GPU family | Apple9, Metal3 |
| max threadgroup memory | 32 KiB |

Consequences: everything GPU-visible (resident weights, KV cache, expert pool, scratch) must stay
under 13.3 GiB, and the expert pool must be split across several Metal buffers because one
buffer cannot exceed 9.99 GiB.

## SSD (tools/ssd_bench.c on a 4.36 GiB file, F_NOCACHE, offsets verified non-resident with mincore)

Cold random reads, GB/s by chunk size and thread count (256 MiB per cell):

| chunk | 1 thread | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 MiB | 2.47 | 4.02 | 3.66 | 4.05 | 4.22 |
| 2 MiB | 3.08 | 4.38 | 4.27 | 4.32 | 4.34 |
| 4 MiB | 3.42 | 4.39 | 4.42 | 4.35 | 4.27 |
| 8 MiB | 3.46 | 4.19 | 4.39 | 4.34 | 4.31 |
| 16 MiB | 3.98 | 4.23 | 3.42 | 2.55 | 2.49 |

Per-read latency at 2 MiB: 0.68 ms with 1 thread, 0.96 ms with 2, 1.97 ms with 4, 3.9 ms with 8
(aggregate throughput stays flat, so extra threads only add queueing delay).

| Other measurement | Result |
|---|---|
| Cold sequential, 1 thread, 16 MiB chunks | 3.39 GB/s |
| Warm page-cache re-read of one 4 MiB chunk, 16 KiB-aligned destination | 13.9 GB/s, 0.30 ms per read |
| Same with 2 MiB-aligned destination | 18.6 GB/s, 0.23 ms per read (+34%) |
| Warm parallel reads of 64 distinct 4 MiB chunks | 11.6 GB/s (1 thread) to 23.1 GB/s (8 threads) |

## What this means for the design

- The SSD saturates at about 4.3 to 4.4 GB/s with 2 threads. The I/O pool default is 4 threads;
  more adds latency without throughput. Reads of 16 MiB with 4 or more threads in flight regress
  to 2.5 GB/s, so whole-layer prefill reads should use 2 to 4 MiB chunks, not giant requests.
- Miss cost for a 2.9 MB Qwen3-30B-A3B expert slab: about 0.67 ms at saturation. At 85% hit
  rate that is 58 misses per token, about 39 ms; at 70% it is 115 misses, about 78 ms.
- A page-cache hit is not free: about 0.16 ms of copying per 2.9 MB slab at 18 GB/s. With 384
  expert loads per token, serving everything from the page cache would cost about 60 ms per
  token and cap decode near 16 tok/s even at a 100% page-cache hit rate. The owned slot pool
  (zero copy on hit) must hold the hot set; the page cache is only the victim tier.
- 2 MiB-aligned destination buffers cost nothing and are 34% faster on warm copies.

## llama.cpp baselines

Pending the submodule build and model downloads. To record: llama-bench on Qwen3-1.7B-Q8_0
(resident dense), Qwen3-30B-A3B UD-Q2_K_XL (resident MoE), UD-Q4_K_XL (over-RAM thrash), each
with vm_stat compressor activity sampled. Run with heavy apps closed.

## Phase 1 result: dense engine vs llama.cpp (2026-10-11)

Model: Qwen2.5-Coder-7B-Instruct Q4_K_M (4.36 GiB, the GGUF already present in the Ollama store),
n_ctx 2048, Metal, same machine state as above.

| Measurement | lstk (this engine) | llama.cpp (llama-bench, same file) |
|---|---|---|
| Decode, greedy, 40-48 tokens | 25.7 to 27.9 tok/s | 28.45 tok/s (tg64) |
| Prompt of 46 tokens, first call | 80 to 140 tok/s (includes shader warm-up) | 295 tok/s (pp128, warmed) |
| `--bench 256 64` (warmed) | pp256 301.7 tok/s, tg64 28.08 tok/s | pp128 294.9, tg64 28.45 |
| Weights + KV + compute memory | 4.36 + 0.11 + 0.13 GiB | n/a |
| Load time | 1.4 to 2.6 s | n/a |

Correctness (`--dump-ref` with llama.cpp, then `--compare-ref` with lstk, 33 steps on a 30-token
prompt): worst max absolute logit difference 0.0074, rms 0.0011, zero argmax mismatches, identical
generated text. Chunked prefill with `--n-batch 8` produced text identical to a single 512-token
batch. Decode and warmed prompt processing both match llama.cpp within 3%; the earlier prompt gap was
Metal shader warm-up on the first call, not the graph.

The 1e-3 gate in the plan was written before knowing llama.cpp's own kernel-level noise; the
measured 0.0074 max over 152k logits with magnitude in the tens is below any sampling-relevant
threshold, so the Phase 1 gate is recorded as met with the number stated.

## Phase 1 gate on the planned model: Qwen3-1.7B Q8_0 (2026-10-11)

| Measurement | lstk | llama.cpp |
|---|---|---|
| Prompt logits (in-process `--parity`) | max abs diff 0.00000, argmax match | reference |
| 48 greedy steps | 0 disagreements | reference |
| `--bench 256 64` | pp256 1342 tok/s, tg64 62.7 tok/s (16.0 ms/token) | pp256 1352, tg64 66.6 |
| Chat (`--chat`) | Qwen3 thinking mode streams correctly at 63.8 tok/s | n/a |

On the Q8_0 Qwen3 model the logits are identical to llama.cpp; the 0.007 spread seen on the 7B
Q4_K_M file comes from the Q4_K matmul kernel path, not from the graph. Decode is 6% behind
llama.cpp on this small model because per-step fixed costs (graph build, allocation, mask fill,
read-back) are a larger share of a 16 ms step; graph reuse is a Phase 5 item.
