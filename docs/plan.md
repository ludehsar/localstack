# Mac-native MoE inference engine for an 18 GB Apple Silicon laptop

Research findings, design, and phased plan. Written 2026-10-11 for the empty project at
`/Users/rashedul/Projects/MyApps/localstack`.

## Status (living copy, updated as phases complete)

- 2026-10-11 Phase 0: Metal cap 13.32 GiB, SSD 4.3 GB/s at 2 threads, page-cache hit copies 0.16 ms per expert. See docs/baseline.md.
- 2026-10-11 Phase 1: dense Qwen2/Qwen3 engine on ggml+Metal; 27.9 tok/s vs llama.cpp 28.45 on a 7B Q4_K_M; logits within 0.0074 of llama.cpp, zero greedy mismatches. Build: `cmake -B build && cmake --build build --target lstk`.
- 2026-10-11 Phase 2: block-allocated KV cache with per-sequence cell ownership (32-token blocks, mask built from ownership), sampler (penalties, temperature, top-k, top-p, min-p), `--chat` REPL using the model's chat template with suffix-only feeding; verified on the 7B (two-turn chat fed 17 new tokens on turn two, KV bytes match formula, parity unchanged).
- 2026-10-11 Phase 3 code: qwen3moe architecture (router, argsort top-k, normalized weights, mul_mat_id), `--trace-routing`, `tools/cache_sim.py`. Validated on a synthetic 2-layer/8-expert GGUF (`tools/make_tiny_moe.py`). Real trace on Qwen3-30B-A3B UD-Q2_K_XL: LRU 86.5% hit at 48 slots per layer, Belady 92%; LFU-decay never wins.
- 2026-10-11 Phase 4 code: `--stream-experts` with per-type-class slot pools wrapped zero-copy as Metal buffers, per-layer graph segments, LRU as an intrusive free list, pinning, pread thread pool, free-memory-aware auto sizing. `tools/test_streaming.sh` proves streaming logits identical to resident logits under forced evictions (3/3 pass). Real model (Qwen3-30B-A3B UD-Q2_K_XL, 11.8 GB, machine with 0.1 GiB free): logits identical across pool sizes; 8.2 / 10.9 / 12.2 / 14.6 tok/s decode at 1 / 2 / 3 / 4 GiB pools (73 to 88% hit), prefill 40 to 50 tok/s. Phase 4 gate (8 tok/s) met. Pools are sharded to at most 1024 slots per tensor (Metal `mul_mat_id` limit found by this run).
- 2026-10-11 Phase 6 (first slice): `--serve` OpenAI-compatible server (chat completions with SSE, stop strings, usage, CORS), one request at a time with KV prefix reuse across turns. Continuous batching, radix prefix cache, and `/metrics` remain.
- Next: `tools/moe_check.sh` on the Q4_K_XL target when its download finishes (close heavy apps first so the resident and llama.cpp comparisons run), then Phase 5 sweeps: pool size up to the page-cache crossover, I/O threads, async segment submission.

## Context

**Goal.** Build an LLM inference engine that runs models larger than RAM on a MacBook Pro
M3 Pro with 18 GB unified memory, by keeping the always-used weights resident and streaming
the rest from the SSD with a smart cache. Mixture-of-Experts (MoE) models are the target
because only a few percent of their weights are touched per token. Secondary goals: learn how
inference engines are built, and produce something usable day to day (CLI chat and an
OpenAI-compatible server).

**Why this is possible now.** MoE models such as Qwen3-30B-A3B activate 8 of 128 experts per
layer per token. Only ~1 GB of a ~17 GB expert table is touched per token, and routing has
strong temporal locality, so a cache holding 30 to 40 percent of the experts serves 80+ percent
of lookups (measured by the community on Qwen3.6-35B-A3B). Apple's unified memory means an
expert read from disk by the CPU is immediately usable by the GPU with no copy.

**Your machine (measured this session).**

| Item | Value | Consequence |
|---|---|---|
| Chip | Apple M3 Pro, 11-core CPU (5P+6E), 14-core GPU, Metal 4 | GPU is memory-bandwidth bound, not compute bound |
| Memory | 18 GB unified, 150 GB/s | Decode ceiling ≈ 150 GB/s ÷ active bytes per token |
| Metal GPU working-set limit | not measured yet; expected ≈ 12 GiB (2/3 of RAM). Verify in Phase 0 | Everything GPU-visible (weights, KV, expert pool) must fit under it, or raise `iogpu.wired_limit_mb` with sudo |
| Page size | 16 KB | Buffers for zero-copy GPU use must be 16 KB aligned; mmap faults are per 16 KB page |
| SSD | 494 GB Apple Fabric NVMe, 74 GB free | Published reads for this model: ~4.5 to 5 GB/s sequential. Measure in Phase 0. Disk space limits which models you can keep |
| OS | macOS 26.6.2 | Metal 4 available; not needed for v1 |
| Current memory state | 11 GB of swap in use right now | Heavy apps must be closed when running the engine; the engine must size itself from free memory, not total |
| Toolchain | Xcode clang 21, CMake 4.4, Rust 1.98, Zig 0.16, Python 3.13, Homebrew. Metal shader toolchain NOT installed (`xcodebuild -downloadComponent MetalToolchain`) | ggml compiles Metal shaders at runtime, so v1 works without it; needed for custom kernels later |
| Existing LLM tools | Ollama with qwen2.5-coder 7B only | llama.cpp must be built from source for baselines |

## Decisions (confirmed by you, 2026-10-11)

| Decision | Choice | Effect on the plan |
|---|---|---|
| Host language and kernel substrate | C++20 + Objective-C++ for Metal calls; ggml via a llama.cpp submodule for kernels, GGUF, tokenizer | Part 4 is final; no kernel-writing phase before streaming |
| First streaming target | Qwen3-30B-A3B (vanilla GQA + MoE, 128 experts, top-8) | Phases 3-5 tune on it; Qwen3.6-35B-A3B and GPT-OSS move to Phase 7 |
| Version 1 interface | CLI chat plus an OpenAI-compatible server | Phase 6 stays in scope |
| Involvement | I implement everything and explain decisions as we go | No user-written components; every phase still ends with a verification you can run |

## Part 1. How an inference engine works (the model you need in your head)

- **Two phases per request.** *Prefill* pushes the whole prompt through the model at once
  (big matrix-matrix products, compute bound). *Decode* produces one token at a time (matrix-
  vector products, memory bound). Every decode step reads every active weight once.
- **The decode speed law.** tok/s ≈ memory bandwidth ÷ bytes of active weights per token. On
  150 GB/s, a 4-bit 7B dense model (≈4 GB) caps at ~37 tok/s; Qwen3-30B-A3B (≈1.9 GB active at
  4-bit) caps at ~80 tok/s. When weights come from SSD instead of RAM, replace 150 GB/s with
  the SSD's ~4.5 GB/s for the missing bytes. This single formula explains every number below.
- **KV cache.** Attention needs the keys and values of all previous tokens. They are stored per
  layer per token. vLLM's PagedAttention stores them in fixed blocks (like OS pages) with a
  per-sequence block table, so memory is not wasted on padding and blocks can be shared.
  SGLang's RadixAttention puts the blocks in a prefix tree so a repeated system prompt or an
  earlier chat turn is never recomputed.
- **Scheduler.** Decides each step which requests run (continuous batching: new requests join
  a running batch at token granularity), how much prefill to chunk in, and what to evict under
  memory pressure.
- **Quantization.** Weights stored at 4 bits instead of 16 cut bytes per token 4x, which is
  directly 4x decode speed on a bandwidth-bound machine. GGUF (llama.cpp's format) is the
  lingua franca: K-quants (Q4_K_M), I-quants (IQ3_XXS, IQ4_XS), and MXFP4 for GPT-OSS.
- **MoE routing.** Each layer has N expert feed-forward networks and a tiny router. The router
  scores the token's hidden state, the top-K experts run, outputs are combined with the router
  weights. Routing is only known after attention of that layer finishes, which limits how far
  ahead expert loads can be issued.
- **Speculative decoding.** A cheap drafter proposes k tokens, the big model verifies them in
  one forward pass. Pays off only when verification is nearly free, which is false when every
  extra token drags its own experts from SSD (measured: break-even or worse for streaming MoE).

## Part 2. What the studied systems do, and what transfers

### Colibri (JustVugg/colibri) — the system you asked about most

**What it is.** A ~145k-line C engine (plus CUDA, Vulkan, a Metal `.mm` backend, and ~124k lines
of Python for conversion and an OpenAI gateway), one monolithic `.c` file per model family
(`c/colibri.c` for GLM-5.2 is 14k lines, `c/deepseek_v4.c` 21k). Safetensors only, no GGUF.
Targets very large MoEs (GLM-5.2 744B, DeepSeek V4, Kimi K3, Qwen3.6-35B-A3B, Qwen3-Coder-30B,
OLMoE). Apache-2.0, extremely active (v2.0.0 on 2026-10-06), heavily AI-assisted, with docs
that drift from code in places.

**How it runs MoE on small machines (with file references):**
- *Classification by tensor name*: routed experts (`mlp.experts.E.{gate,up,down}_proj`) stream;
  everything else is resident int4/int8 with dequant on use (`c/colibri.c:3063-3086`, `407-429`).
- *Unit of caching*: one slab per (layer, expert) holding all three matrices, read with one
  coalesced `pread` when the shard stores them contiguously (`c/colibri.c:3140-3315`). On Metal,
  slabs are `posix_memalign(16384)` and wrapped zero-copy (`c/backend_metal.mm:1022-1041`).
- *I/O*: `pread` into owned slabs, page cache deliberately left on as a free second tier
  (`DROP=0`, `c/colibri.c:1371`). `DIRECT=1` uses `F_NOCACHE` on macOS (`c/compat.h:54-63`).
  Their own Mac probe: fast SSD (≥4 GB/s) + Metal → default expert cache of **one slot per
  layer**, "page cache favored" (`c/colibri.c:12454-12467`). Reason: every slab is its own
  `MTLBuffer` re-declared with `useResource:` per command buffer, which stalled the GPU for
  seconds at cap 16 (`docs/experiments/e5-metal-residency-set.md`). On an M1 Ultra that default
  cost 0.98 vs 1.45 tok/s against cap 33.
- *Budget*: 88% of available memory minus resident bytes, 1.2 GB activations, a 2.5 GB
  page-cache reserve, 64 widest experts, KV at max context; refuses to start if a single slot
  per layer does not fit; an RSS guard every 16 tokens shrinks capacity in place
  (`c/colibri.c:11780-11916`, `9795-9847`).
- *Cache structures*: per-layer slot arrays plus a direct `slot_by_expert[layer][eid]` table
  (O(1) lookup that re-checks identity), exact LRU via a global atomic clock with an O(cap)
  victim scan that skips `in_flight` slots (`c/colibri.c:435-474`, `505-506`). The cleanest
  variant is DeepSeek V4's `ColiExpertStore` vtable with refcounted leases, intrusive per-layer
  doubly-linked LRU, and in-flight dedupe (`loading_expert` + condvar) so two requests for the
  same expert issue one read (`c/expert_store.h:40-69`, `c/deepseek_v4.c:9454-9691`). The Qwen3.6
  "qpack" pool: fixed pool of 96 16 KiB-aligned blobs registered with Metal once, addressed by
  offset, with generation-checked references (`c/qwen36_qpack.c:1-37`, `139-169`).
- *Prefetch*: `PIPE` overlaps miss reads with compute of the resident experts inside a layer
  (8 pthreads), default on only on Windows. `PILOT` router lookahead applies layer L+1's norm and
  router to layer L's post-attention residual and prefetches the top 6-8; claimed recall 71.6%
  versus 41.3% for "same experts as last token" (`c/colibri.c:7203-7265`, `docs/tuning.md:184`).
  `COUPLE` adds an offline co-activation table. All off by default on macOS; the M1 Ultra report
  shows decode "fully serial" (disk wait 57%).
- *Miss handling*: stall by default; opt-in lossy modes restricted to decode (`DEGRADE_ZERO`
  with no renorm at gate < 0.03, `EXPERT_BUDGET`, cache-aware routing).
- *Compute*: all own kernels (AVX2/VNNI/AMX/NEON), no ggml/BLAS; expert kernel for Qwen has
  no NEON path, so Apple CPUs run scalar code (`c/expert_ffn.h:176-267`). Metal MoE kernel is
  "bindless" (GPU addresses per expert) but scatter-adds on the CPU, allocates MTLBuffers per
  call, and resolves addresses with a linear scan under a mutex (`c/backend_metal.mm:1052`).
- *Runtime*: serial prefill, continuously batched decode across up to 16 KV slots (batch-union
  means each expert is read once for all rows), MTP and prompt-lookup speculation with an
  adaptive gate, GBNF grammar, hand-written tokenizers and chat templates.
- *Measured*: GLM-5.2 744B 1.25→2.24 tok/s on M5 Max 128 GB (Metal); Qwen3-Coder-30B int4
  8.5-9.6 tok/s all-in-RAM and 5.1 tok/s at cap 32 (6.5 GB resident) on their CPU path. Their
  own offline simulator says decayed-frequency admission would cut felt wait 40-66% over LRU;
  never shipped.

**Transfer verdict.** Steal the *ideas*, not the code: fixed-stride 16 KiB container + fixed
pool registered once with generation-checked refs; lease-based store with in-flight dedupe;
honest budget with reserves, refuse-to-start, and RSS guard; batch-union MoE with
"resident first, misses second"; router lookahead as an opt-in experiment; a route-trace
simulator before writing policy code. Avoid: per-slab Metal buffers, env-var sprawl, five
parallel cache implementations, and `mlock` of pins on an 18 GB machine.

### SGLang (sgl-project/sglang)

Three processes over ZMQ (tokenizer, scheduler, detokenizer). The parts worth copying:
- *Radix prefix cache* (`python/sglang/srt/mem_cache/radix_cache.py:250-797`): page-granular
  radix tree whose node values are KV slot ids; `lock_ref` along root paths protects KV in use;
  an evictable-leaf set and (in `unified_cache/unified_tree_core.py:212-388`) an intrusive LRU
  list with a logical clock make eviction O(1); matching uses exponential+binary search over
  C-level slice compares; a request's duplicate KV is freed on insert when another request got
  there first; chunked prefill checkpoints finished chunks into the tree.
- *Paged KV* (`mem_cache/memory_pool.py:373-499`, `allocator/paged.py`): `req_to_token[req,
  pos] → slot`, page 0 and row 0 as padding sinks, free list of page ids with lazy sort.
- *Scheduler* (`managers/scheduler.py:3682-4346`): prefill-first loop, `PrefillBudget =
  available + evictable − reserved` with a `new_token_ratio` that decays over 600 steps and
  resets after a retraction, chunked prefill, retraction least-progress-first, priority preemption.
- *Overlap scheduling*: CPU schedules step N+1 while the GPU runs step N; sampled tokens are
  relayed through a slot-indexed GPU buffer (`FutureMap`, `managers/overlap_utils.py:255-329`)
  so the next step launches without reading tokens back; finish detection lags one step. The
  new MLX backend chains lazy graphs the same way (`hardware_backend/mlx/scheduler_mixin.py`).
- *MoE*: `moe_align_block_size` sorts (token, k) pairs by expert so each expert's weights are
  read once per block; padded rows get `topk_ids = -1` and never trigger an expert
  (`layers/moe/topk.py:1736-1752`); the KTransformers wrapper masks non-resident experts to -1
  for the GPU and computes them elsewhere; `eplb/expert_distribution.py:614-626` keeps
  per-layer expert counts via `scatter_add` into a ring of recent passes.
- *HiCache*: a disk tier for KV keyed by chained SHA-256 page hashes, write-through-selective
  (≥2 hits), per-layer completion events so layer i computes while layer i+1 loads
  (`mem_cache/l2_transfer.py:144-179`).

**Transfer verdict.** Radix cache, paged KV with padding sinks, the budget/retraction logic,
one-step overlap with a token relay, the expert-count ring, and the -1 masking rule transfer
almost verbatim. Do not copy the separate host KV tier (same DRAM on a Mac), the ZMQ process
split, the dozens of policies, or any TP/DP/EP plumbing.

### vLLM (vllm-project/vllm)

vLLM has **no expert cache and no SSD weight streaming anywhere** (its GGUF loader is gone in
this snapshot too). Its value for us is the quality of its memory-management primitives:
- *Block pool as LRU* (`vllm/v1/core/block_pool.py`, `kv_cache_utils.py:175-507`): an
  intrusive doubly-linked free list threaded through the blocks themselves; a cached block with
  `ref_cnt == 0` stays in the free list *and* in the hash map and is evicted lazily only when
  reused; `touch()` unlinks a hit in O(1); frees go tail-first so a prefix chain's root outlives
  its tail; unhashed blocks are prepended (reuse first), hashed appended (evict last). The
  null block (id 0) absorbs writes for padded positions.
- *Prefix hashing* (`kv_cache_utils.py:625-725`): `hash(parent_hash, block_tokens, extra_keys)`,
  full blocks only, recomputed as tokens append; the last token is always recomputed to get
  logits; lookup stops at the first miss because each hash covers its whole prefix.
- *Scheduler* (`vllm/v1/core/sched/scheduler.py:591-1567`): no prefill/decode phases, just
  "make `num_computed_tokens` catch up to `num_tokens_with_spec`" under a token budget;
  all-or-nothing `allocate_slots`; the whole prompt must fit before admission; preemption is
  recompute (free blocks, requeue at front), and the freed hashed blocks usually re-hit;
  `SchedulerOutput` carries diffs, not state; grammar bitmasks are built while the GPU runs.
- *Memory arithmetic* (`kv_cache_utils.py:1003-1227, 1520-1687`): `num_blocks = available //
  bytes_per_block`, one block reserved for null, startup fails unless one request at
  `max_model_len` fits, and a binary search reports the maximum length that would.
- *MoE* (`model_executor/layers/fused_moe/`): `expert_map` marks non-local experts `-1` and the
  kernel writes zeros for them; a no-sort decode path when `tokens × topk × 4 ≤ n_experts`
  (every (token, k) pair is its own block, exactly single-user decode); padding-free counting
  sort for prefill; routing weight applied in fp32 before the downcast; per-expert atomic load
  counters feed EPLB.
- *Prefetch offloader* (`model_executor/offloader/prefetch.py`): a ring of `prefetch_step` GPU
  slots per tensor shape with fork/done events so a slot is never overwritten while the previous
  layer may still read it. Routing-blind and cache-less, but the fence pattern is right.
- *CPU KV offload tier* (`vllm/v1/kv_offload/cpu/policies/arc.py`): ARC (recency list T1,
  frequency list T2, ghost lists) plus an admission threshold (store only keys seen ≥ 2 times).
  Scan-resistant, which matters for us because a prefill touches nearly every expert once.
- *Speculation* (`vllm/v1/spec_decode/ngram_proposer.py`, `sample/rejection_sampler.py`):
  prompt-lookup n-gram drafting needs no draft model; greedy and p/q rejection with recovered
  and bonus tokens; rejected KV is overwritten later, never freed.
- *Sampler ordering* (`vllm/v1/sample/sampler.py`): fp32 → masks → argmax-changing processors
  → penalties → (greedy shortcut) → temperature → min-p → top-k/top-p → sample.

**Transfer verdict.** Block-pool-as-LRU (reused for the expert slot pool), chained prefix
hashing, the unified token-budget scheduler with recompute preemption, indirection tables as
the only per-step variation, the `-1` expert mask, ARC/admission for scan resistance, n-gram
drafting with the rejection sampler, and the sampler order. Do not copy: the separate EngineCore
process and ZMQ (a GIL workaround), async-scheduling placeholders, dummy-forward memory
profiling (we own every allocation), UVA `cpu_offload_gb`, and the N-group hybrid machinery.

### Mac-specific prior art (read in full this session)

| System | What it is | Measured result | Lesson for us |
|---|---|---|---|
| Flash-MoE (danveloper, Mar 2026) | 7k lines Objective-C + 1.3k lines Metal, from scratch. Qwen3.5-397B on M3 Max 48 GB | 4.4 tok/s at 4-bit, 5.7 at 2-bit. 90 experiments logged | Same OS generation as yours. `pread` beats mmap 5x for cold data (16 KB page faults). Its wired Metal LRU cache (10-21 GB) lost to the OS page cache by 38% because wired memory drove the memory compressor to 60-130k decompressions/s. Kernel hints (`F_RDADVISE`, `MADV_*`) neutral or harmful. 2 MB-aligned destination buffers: 3.6x faster warm copies. SSD DMA during GPU compute slowed the GPU 73% (shared memory controller). Prediction from previous token: 25% hit on 512 experts K=4, net loss |
| Flash-MoE Anemll fork | Adds llama.cpp GGUF dequant kernels (IQ3_XXS/IQ4_XS/Q5_K), persistent 8-thread I/O pool, page-aligned read fan-out (`--cache-io-split 4`), perplexity harness | 12.9 tok/s on M5 Max; +10% from read fan-out; Q3 experts 23% less I/O for PPL 3.81 vs 3.64 | Vendoring ggml's quant kernels works. Expert compute is ~2% of time; dense matvec bandwidth and expert I/O are everything. A PPL harness is mandatory for every optimization |
| mlx-lm issue #1438 (community study, 2026) | Expert offload prototypes in MLX on 32 GB, 48 GB, 128 GB Macs. Qwen3.6-35B-A3B, Qwen3-235B, GLM-5.2 | Qwen3.6 4-bit: 57 tok/s resident, 10.8 tok/s streaming at 30% residency. Steady-state hit rate 0.80-0.87 at 30% residency; Belady ceiling 0.90-0.94 | Closest study to our target. Plain per-layer LRU wins; pinning and predictive prefetch lose (adjacent-layer Jaccard 0.017 vs 0.016 random). Per-expert contiguous layout +14-18% decode. 2 parallel miss reads +30%, more hurts when warm. **Cache sizing has a crossover: past it the resident set starves the page cache (60 GB cache was 65% slower than 35 GB)**. Prefill drops to ~77 tok/s under streaming |
| **llama.cpp discussion #23324 (Metal PoC, same chip as yours)** | A llama.cpp fork: per-layer compact pool of N expert slots in Metal shared memory, a tiny Metal interceptor kernel publishes the routed ids, a CPU sidecar resolves LRU + `pread` from the unmodified GGUF, signals an `MTLSharedEvent`, and `MUL_MAT_ID` runs unchanged on the compact pool. Exact inference | Qwen3-30B-A3B Q6_K on **M3 Pro 36 GB**: vanilla 38.1 tok/s; 32 slots/layer (10.6 GiB wired) 22.1 tok/s; 16 slots (7.8 GiB) 15.7; 8 slots (6.3 GiB) 12.0. Author reports ~13 tok/s on an M1 Pro 16 GB | **Existence proof of this plan's core design on your exact chip.** Plain LRU; an SLRU variant gave nothing. Constraint `ub × n_expert_used ≤ n_slots` for batching. Sets the calibration for our targets: 12-20 tok/s on Qwen3-30B-A3B at Q4 within ~8 GiB wired |
| mlx-flash (szibis) | Python + MLX + C GCD cache sidecar, many claimed techniques | Claims 82 tok/s on Qwen3-30B-A3B under memory pressure; LCP cache 70.7% hit; cross-layer co-occurrence predictor | Its "expert streaming" = GPU lookup table mapping expert id → cache slot + pre-stacked tensors + `gather_qmm`, cache updated between tokens. Exactly our slot-pool design, validates it on MLX. Its fallback drops missing experts (lossy); we will not |
| llama.cpp today | `--moe-cache-mib` GPU LRU of host experts merged Oct 7 2026 (CUDA/HIP, GPU↔host split, useless on unified memory); mmap `--lazy-mode`; discussion #27149 proposes SSD expert streaming (prototype only) | #29887: 1.6-2.2x TG at 72-89% hit on CUDA. #27149: Qwen3-30B-A3B full layer 345 MB vs 8 experts 30 MB; 15.75 GB experts, 0.48 GB shared+router | Nobody ships RAM↔SSD expert streaming for Mac in llama.cpp. mmap demand paging is layout-blind and faults per 16 KB page. llama.cpp maps whole GGUF files as one Metal buffer, so even CPU-only tensors count against the GPU working set (issue #29465) |
| Papers | LLM in a Flash (Apple): windowing + row-column bundling. PowerInfer-2: neuron clusters, I/O pipelining. HOBBIT: low-precision fallback on miss, built on llama.cpp. MoE-Infinity: LFU on activation traces. AdapMoE/Pre-gated: next-layer gate prediction (86% recall on their models) | | Bundling = our per-expert slab. Prediction results on Qwen3.x do not reproduce the papers' recall; treat prefetch as an experiment, not a design pillar |

## Part 3. Apple Silicon facts that drive the design

1. **One memory pool.** CPU, GPU, and SSD DMA share the same DRAM and memory controller. A
   `pread` into a page-aligned host buffer wrapped with `newBufferWithBytesNoCopy` is a GPU
   buffer with zero copies. There is no GPU↔host tier; the only tier below RAM is the SSD.
2. **Two kinds of RAM pages.** *Wired* pages (Metal buffers while in use) cannot be compressed
   or evicted. *File-backed page-cache* pages (what `pread` without `F_NOCACHE` leaves behind)
   are evicted first and cost nothing to drop. An app-level cache made of wired memory squeezes
   the page cache and triggers the compressor. Rule from Flash-MoE and mlx-lm: keep the owned
   cache well below "as big as fits" and let the page cache act as the victim tier.
3. **GPU working-set cap.** Metal reports `recommendedMaxWorkingSetSize` (≈12 GiB expected
   here). Exceeding it makes the driver page GPU memory. Page-cache pages are not GPU-visible
   and do not count, so on this machine the page cache can use memory the GPU cannot.
4. **Decode is bandwidth bound even when resident.** The 14-core GPU saturates ~150 GB/s on
   dequant-matvec kernels. Background SSD DMA competes on the same controller; Flash-MoE
   measured a 73% GPU slowdown during overlapped prefetch. Expect serial GPU→SSD→GPU per layer
   to be near-optimal; test overlap, do not assume it.
5. **I/O API.** No io_uring. Use a persistent pool of 2-8 threads issuing `pread` of whole
   expert slabs (one syscall, one DMA each). mmap only for the resident tier. `dispatch_io`
   was 70% slower, `aio_read` 7% slower than a thread pool in Flash-MoE's tests. Destination
   buffers 2 MB aligned.
6. **CPU side.** Accelerate (AMX) is the right tool for small recurrent state updates (Gated
   DeltaNet in Qwen3.5/3.6). The Neural Engine is not usable for dynamic MoE routing.
7. **Metal 4** (MTLTensor, ML encoders) exists on this OS but targets fixed graphs; ggml's
   Metal kernels are the practical path for v1.

## Part 4. Language and substrate decision

**Recommendation: C++20 for the engine, Objective-C++ (`.mm`) for the few Metal API calls,
Metal Shading Language for any custom kernels, and ggml (vendored via a llama.cpp submodule)
as the compute substrate.**

Why C++ wins here:
- Every serious Apple Silicon inference codebase (ggml/llama.cpp, MLX, whisper.cpp, Flash-MoE)
  is C/C++/Objective-C. Reuse and learning-by-reading are maximized.
- Metal's native API is Objective-C with Apple's official `metal-cpp` bindings; MSL kernels are
  a C++ dialect, so one mental model covers host and device code.
- Flash-MoE rewrote from MLX/Python to C/Metal and got most of its gains from direct command
  buffer control; mlx-flash needed a C sidecar for the same reason.

Why ggml rather than kernels from scratch: the novel work in this project is the memory tier,
the cache, the I/O scheduler, and the server. Those are what vLLM, SGLang, and Colibri are
famous for; none of them wrote their own GEMMs. ggml provides (verified against current headers):
`ggml_backend_dev_buffer_from_host_ptr` (zero-copy Metal buffer over our page-aligned pool),
`ggml_mul_mat_id` (fused gathered MoE matmul over a 3D expert tensor with per-token ids),
`ggml_flash_attn_ext`, `ggml_set_rows` (paged KV writes), `ggml_add_id` (per-expert bias for
GPT-OSS), every GGUF quant type including MXFP4, events and async graph compute, plus llama.cpp's
tokenizer and chat-template code. A custom Metal kernel for the expert gather is a Phase 7
option once profiling says ggml's is the bottleneck (Anemll's data says it is not).

Alternatives considered:
- **Rust** (`objc2-metal`): best language for the concurrency-heavy cache and I/O code, but
  Metal bindings are thin and `unsafe`, no mature Metal inference stack to read, and ggml via
  FFI adds a boundary for a solo learner. Credible second choice, not the default.
- **Swift**: first-class Metal, but weak for systems-level buffer management and no low-level
  inference ecosystem.
- **Zig** (installed): clean C interop, but Metal is Objective-C; calling it through
  `objc_msgSend` is painful and the community is tiny.
- **Mojo/MAX**: Apple GPU support arrived in 25.6/26.4 and runs some models, but it is a
  closed model-serving framework; you cannot build a custom memory tier inside it.
- **MLX C++** as substrate instead of ggml: clean API and `gather_qmm`, but MLX owns its
  allocator, so wrapping our pread buffers zero-copy is not supported, and GGUF coverage is
  weaker. Keep as fallback.

## Part 5. Design

### 5.1 Components (bottom to top)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ API: CLI chat · OpenAI-compatible HTTP (/v1/chat/completions, SSE)      │
├──────────────────────────────────────────────────────────────────────────┤
│ Scheduler: sessions, continuous batching, radix prefix cache, sampling   │
├──────────────────────────────────────────────────────────────────────────┤
│ Model graphs (per architecture): attention+router graph │ expert graph   │
├───────────────────────────┬──────────────────────────────────────────────┤
│ KV cache (paged blocks)   │ Expert store: slot pool · policy · I/O pool  │
├───────────────────────────┴──────────────────────────────────────────────┤
│ Memory budget: resident tier (mmap) · GPU working-set accounting         │
├──────────────────────────────────────────────────────────────────────────┤
│ Formats: GGUF reader (gguf.h) · repacked expert slabs (own format)       │
├──────────────────────────────────────────────────────────────────────────┤
│ ggml + Metal backend (kernels, graph exec) · llama-vocab · minja         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tensor classification and on-disk layout

- **Resident tier**: embeddings, output head, norms, attention projections, routers, shared
  experts, Gated DeltaNet weights. Loaded once from the GGUF into one page-aligned buffer,
  wrapped as a Metal buffer. Only these tensors are mapped for the GPU (unlike llama.cpp, which
  maps the whole file).
- **Streamed tier**: the `ffn_{up,gate,down}_exps` 3D tensors. A one-time `repack` tool writes
  `experts/layer_NN.bin` where expert *e* is one contiguous slab at offset `e * slab_bytes`
  holding up, gate, down (quant blocks kept byte-identical to the GGUF), slab size rounded up
  to 16 KB, plus `manifest.json` (dims, types, offsets, source GGUF hash). One expert = one
  `pread`. (Qwen3-30B-A3B Q4_K: ≈2.9 MB per slab; Qwen3.6-35B-A3B: ≈1.9 MB; GPT-OSS: 13 MB.)

### 5.3 Expert store (the heart)

- **Slot pool.** `n_slots` fixed slabs in a few large page-aligned allocations (2 MB alignment
  per slot), each allocation wrapped **once** as a Metal buffer and never re-registered (Colibri's
  per-slab `MTLBuffer` + `useResource:` pattern is what forced its cap-1 default on Macs). ggml
  sees three 3D tensors `[n_slots, out, in]` (up/gate/down views at slab-internal offsets) that
  never change shape; slots are addressed by index.
- **Tables.** `slot_of[layer][expert] → slot or NONE` (flat array, O(1)); `owner_of[slot] →
  (layer, expert, generation)`; per-slot state `FREE / LOADING / READY / IN_USE(refcount)`;
  per-slot `last_used` tick and decayed `heat` counter. References carry `(slot, generation)`
  and are re-checked at dispatch so a refilled slot is refused rather than silently used.
  A miss on an expert already `LOADING` waits on that load instead of issuing a second read.
- **Lookup per layer per step.** After the router runs, the CPU reads the top-K ids, maps them
  to slots, issues `pread` for misses into victim slots chosen by the policy, waits, then writes
  the slot ids into the `ids` tensor and runs the expert graph. In-use slots are pinned until
  the GPU finishes the layer.
- **Policy.** v1: global LRU implemented as vLLM's block pool (an intrusive free list where
  READY-but-unreferenced slots sit at the tail and are evicted lazily on reuse; a hit unlinks in
  O(1)). Candidates for Phase 5, chosen by offline simulation on recorded routing traces: LFU
  with exponential decay plus admission check (community trace sims beat LRU), ARC or S3-FIFO
  for scan resistance, and per-layer capacity shaping.
- **Prefill does not pollute the pool.** A prompt touches nearly every expert once per layer;
  under LRU that flushes the decode working set. Prefill reads misses into a separate scratch
  region (sequential whole-layer read when the union is large) and only *touches* pool hits;
  the pool keeps the decode-hot set.
- **Miss handling.** Exact by default: stall for the read. Optional lossy mode (drop missing
  experts and renormalize router weights) only as an explicit flag for experiments.
- **Page cache stays on.** No `F_NOCACHE` in normal operation: evicted slabs survive in the
  page cache as a free victim tier that is not GPU-visible. Pool size is swept to the crossover
  point (Phase 5), starting at 30% of the expert table.
- **I/O pool.** Persistent threads (default 4, swept 1-8), one `pread` per slab, optional
  page-aligned split into 2-4 chunks for warm reads. Completion flips slot state with an atomic.

### 5.4 Per-token data flow (decode, one sequence)

```
for layer in layers:
    G1: norm → attention (flash_attn_ext over paged KV) → o_proj → residual → norm → router logits
    CPU: top-K + weights (and shared-expert mask)       ~ microseconds
    store.ensure(layer, topK) → misses via I/O pool     ~ 0 ms on hit, ~0.6 ms per miss at 4.5 GB/s
    G2: mul_mat_id(up/gate on pool, ids=slots) → SwiGLU → mul_mat_id(down) → weighted sum → residual
final: norm → output head → logits → sampler
```
v1 runs G1 and G2 as separate ggml graphs per layer (one CPU round trip each, ~0.1 ms). If the
per-layer sync shows up in profiles, Phase 5 submits G2 asynchronously and overlaps the next
layer's G1 encode with it (Flash-MoE's deferred-command-buffer pattern via
`ggml_backend_graph_compute_async` and events), or moves to the #23324 pattern where the GPU
waits on an `MTLSharedEvent` signalled by the I/O sidecar instead of a CPU round trip.

### 5.5 Prefill

Batch the whole prompt. Per layer, compute the union of experts needed by all prompt tokens; if
the union exceeds ~40% of the layer, read the whole layer's expert file sequentially into a
scratch region (one large read at full SSD speed), else read the union. Chunk prompts so the
scratch region fits the budget. Expected 50-80 tok/s prefill under streaming (community
numbers), versus hundreds when resident.

### 5.6 KV cache and prefix cache

Paged KV with 32-token blocks allocated from a fixed pool, per-sequence block tables, f16 by
default with a q8_0 option. A radix tree over block hashes (SGLang style) gives multi-turn and
shared-system-prompt reuse for free. Qwen3-30B-A3B costs 96 KB per token at f16 (32 K context =
3 GB, so q8 or a 16 K default matters). Qwen3.6-35B-A3B costs 20 KB per token plus a fixed 63 MB
recurrent state per sequence.

### 5.7 Memory budget algorithm (startup)

```
gpu_cap     = recommendedMaxWorkingSetSize (or user override)
free_now    = host free + inactive pages (host_statistics64), minus a user reserve (default 3 GiB)
resident    = size of resident tier (known from GGUF)
kv          = n_ctx × bytes_per_token × max_sessions
scratch     = measured by a ggml reserve pass
pool        = min(gpu_cap - resident - kv - scratch,  free_now - resident - kv - scratch - page_cache_floor)
page_cache_floor default 3 GiB; pool clamped to [2 GiB, 45% of expert table]; all overridable
```
Everything is logged at startup with the predicted hit rate from the trace simulator. The engine
refuses to start if the pool would fall below its floor (no silent swapping), and a guard every
16 tokens compares measured RSS and compressor activity against the budget and shrinks the pool
in place if exceeded (Colibri's `rss_guard` idea).

### 5.8 Observability

Per token: ms in G1, G2, I/O wait, sync; misses per layer; bytes read; pool hit rate (rolling
and per layer); `vm_stat` compressor decompressions/s sampled every second; GPU allocated bytes.
Exposed on the CLI `--timing` flag and the server's `/metrics`. Every optimization in Phase 5
must show its number here plus unchanged perplexity.

### 5.9 Expected performance on this machine (to be replaced by measurements)

| Model | File | Fits in 12 GiB GPU cap? | Resident ceiling | Streaming estimate (85% hit, 4.5 GB/s SSD) |
|---|---|---|---|---|
| Qwen3-30B-A3B UD-Q4_K_XL | 17.7 GB | No (thrashes in llama.cpp) | ~80 tok/s theoretical | 384 expert loads/token, ~58 misses × 2.9 MB → 10-15 tok/s by bandwidth math; the #23324 PoC on this chip suggests 12-20 |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 22.4 GB | No | ~90 tok/s theoretical, 57 measured by community | 320 loads/token, ~48 misses × 1.9 MB → ~12-18 tok/s |
| GPT-OSS-20B MXFP4 | 12.1 GB | Barely not (11.3 GiB + KV) | ~50 tok/s | 96 loads/token, 13 MB each → ~8-10 tok/s |
| GPT-OSS-120B MXFP4 | 60.8 GiB | No, and disk is too small without an external SSD | n/a | ~3-4 tok/s at ~45% hit (stretch goal) |

Baseline to beat: llama.cpp with mmap on the same over-RAM file (expected 1-3 tok/s or a
hang; measured in Phase 0). Calibration from the #23324 PoC on an M3 Pro: 12-22 tok/s on
Qwen3-30B-A3B Q6_K with 6-11 GiB wired for experts; Q4 experts are ~35% smaller, so the same
wired budget holds ~50% more experts.

## Part 6. Phased plan

Each phase ends with a verification you can run. I implement every component; design
decisions with real trade-offs (eviction policy, budget split, miss handling, sampler order) are
explained in the chat and recorded in `docs/decisions.md` as they are made. Rough total: 9 to
11 weeks of focused work for Phases 0-6.

### Phase 0. Measure the machine and the baseline (1-2 days)
- Build llama.cpp from source with Metal (`brew install cmake` done; `cmake -B build && cmake --build build -j`).
- Download: Qwen3-1.7B Q8 GGUF (dense, fits; parity testing), Qwen3-30B-A3B UD-Q4_K_XL
  (17.7 GB, the over-RAM target), Qwen3-30B-A3B UD-Q2_K_XL (11.8 GB, fits under the GPU cap for
  MoE correctness work). ~31 GB total of the 74 GB free.
- Read `recommendedMaxWorkingSetSize` with a 6-line Swift script.
- SSD study: cold-read throughput vs queue depth (1-16 threads) and read size (1-16 MB) with
  `F_NOCACHE` and fresh offsets per run; warm-read copy throughput by buffer alignment.
- `llama-bench` on the dense model (resident tok/s vs the 150 GB/s ceiling), on the Q2 MoE
  (resident MoE baseline), and on the Q4 MoE (the thrash number), with `vm_stat` sampled.
- Output: `docs/baseline.md`. Verification: the numbers exist and are reproducible twice.

### Phase 1. Dense skeleton (1 week)
- CMake project; llama.cpp as a git submodule (ggml + `llama-vocab` + minja); Objective-C++
  file for Metal device queries.
- GGUF reader → tensor manifest → resident buffer → ggml tensors.
- Qwen3 dense decoder graph (RMSNorm, GQA attention with RoPE, SwiGLU), greedy decode loop.
- Verification: logits for a 64-token prompt match llama.cpp within 1e-3 max abs error (f32
  KV); tok/s within 15% of `llama-bench`.

### Phase 2. KV cache, sampling, chat (1 week)
- Paged KV pool and block tables; `ggml_set_rows` writes; f16/q8 option.
- Sampler chain in vLLM's order (fp32 logits, masks, penalties, greedy shortcut, temperature,
  min-p, top-k/top-p, sample). Chat template via minja. Streaming CLI with multi-turn reuse.
- Verification: multi-turn chat works; KV bytes match the formula; greedy output identical to
  Phase 1.

### Phase 3. MoE resident (1 week)
- Router (softmax/top-k, `norm_topk_prob`), `mul_mat_id` expert FFN, shared expert, on
  Qwen3-30B-A3B UD-Q2_K_XL fully resident.
- `--trace-routing out.bin` records per-layer top-K ids per token; `tools/cache_sim.py` replays
  traces against LRU / LFU-decay / S3-FIFO at several pool sizes and prints hit rates.
- Verification: parity with llama.cpp on the same file; tok/s near `llama-bench`; a trace of
  2k tokens over 5 diverse prompts with the simulator's hit-rate curve in `docs/`.

### Phase 4. Streamed experts v1: it runs (2 weeks)
- `repack` tool (GGUF → per-layer slab files + manifest, verified byte-for-byte).
- Slot pool, tables, state machine, global LRU as the block-pool free list, persistent `pread`
  pool, per-layer G1/G2 split, the memory budget algorithm of 5.7 with its refuse-to-start floor.
- Run Qwen3-30B-A3B UD-Q4_K_XL within a ~10 GiB engine footprint.
- Verification: on the UD-Q2_K_XL file (which fits), logits are bit-identical between resident
  mode and streaming mode with a deliberately tiny pool (every layer misses); on the UD-Q4_K_XL
  file, top-1 tokens match llama.cpp's slow mmap run over 64 tokens on 3 prompts; hit rate and
  tok/s logged; target ≥ 8 tok/s steady state, no compressor storm in `vm_stat`.

### Phase 5. Make it fast (2-3 weeks, experiment driven)
Each item is an A/B with `--timing` and a 2k-token perplexity check:
- Pool size sweep to the page-cache crossover; I/O threads 1-8; read fan-out 1-4.
- Policy from the Phase 3 simulator's winner.
- Async G2 submission overlapping the next layer's encode; measure GPU contention.
- Prefill union-loading path (5.5); chunk size sweep.
- Optional: lookahead routing on the pre-attention state, kept only if it wins end to end.
- Verification: ≥ 15 tok/s decode and ≥ 50 tok/s prefill on the Q4 target within ~8 GiB of
  wired expert memory (the #23324 PoC reached 15.7 tok/s at 7.8 GiB on Q6_K with plain LRU, so
  this is a floor, not a stretch), perplexity equal to resident within noise.

### Phase 6. Serve (1-2 weeks)
- OpenAI-compatible HTTP server with SSE streaming, sessions, radix prefix cache over KV blocks
  (SGLang's lock_ref + evictable-leaf design), decode continuous batching with padded rows
  masked to expert id -1 so padding never triggers a fetch (SGLang), one-step-ahead scheduling
  with a slot-indexed token relay, memory-aware admission with retraction, `/metrics`.
- Verification: an off-the-shelf chat UI connects; 3 concurrent sessions show aggregate
  throughput above single-session.

### Phase 7. Stretch
- Qwen3.6-35B-A3B (Gated DeltaNet layers via ggml's delta-net ops, Accelerate for state).
- GPT-OSS-120B from an external Thunderbolt SSD.
- Prompt-lookup n-gram drafting with vLLM's rejection sampler (no draft model needed), with k
  adapted to the acceptance rate and measured against the extra expert I/O; a custom Metal
  expert-gather kernel if profiles justify it; an SSD tier for idle sessions' KV (SGLang's
  chained-hash page store).

## Part 7. Risks and how the plan handles them

- **Memory headroom on 18 GB with your current app load.** Phase 4 sizes from free memory and
  refuses to start below the floor rather than swapping.
- **Hit rate on Qwen3-30B-A3B is unmeasured** (community numbers are for Qwen3.6). The Phase 3
  trace study answers it before Phase 4 is built.
- **The calibration machine had more page cache than you do.** The #23324 PoC ran on a 36 GB
  M3 Pro, so ~20 GB of page cache sat behind its small wired pool; on 18 GB the page cache will
  be 3 to 5 GB and the pool must carry more of the hit rate. Expect the low end of the 12-20
  range first; the pool-size sweep in Phase 5 finds this machine's crossover.
- **GPU/SSD contention** may make overlap useless; Phase 5 measures rather than assumes.
- **ggml API churn.** Pin the submodule commit; upgrade deliberately.
- **Disk space.** 74 GB free limits the model set; GPT-OSS-120B needs external storage.

## Verification (end to end)

1. `tools/bench.sh` runs llama-bench on the resident dense model and the engine on the same
   model: engine within 15% (Phase 1 gate).
2. `tools/parity.sh` compares engine logits to llama.cpp on 5 prompts for each supported
   architecture (Phases 1, 3, 4 gates).
3. `tools/stream_bench.sh` runs the Q4 MoE target for 300 tokens on 5 prompts, printing tok/s,
   hit rate, bytes read, compressor rate; must beat the Phase 0 llama.cpp thrash number by 3x
   (Phase 4) and reach the Phase 5 targets.
4. `tools/ppl.sh` computes perplexity on 2k tokens of WikiText-2 resident vs streamed; equal
   within 0.5%.
