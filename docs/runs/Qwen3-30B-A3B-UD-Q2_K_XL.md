# Qwen3-30B-A3B-UD-Q2_K_XL

Run: 2026-10-11 21:10, available memory (free+inactive) at start: 5.0 GiB

## Streaming self-consistency (greedy, pool 1 GiB vs 2.5 GiB)
```
model: /Users/rashedul/Models/gguf/Qwen3-30B-A3B-UD-Q2_K_XL.gguf  layers=48 embd=2048 heads=32/4 head_dim=128 ff=6144 vocab=151936  experts=128 used=8 ff_exp=768
memory: free 0.12 GiB, inactive 3.32 GiB, wired 6.18 GiB, compressed 4.30 GiB, swap used 1.14 GiB
memory: weights 0.95 GiB  kv 0.19 GiB  compute 0.07 GiB  metal working-set cap 13.32 GiB  load 766 ms
prompt: 18 tokens in 1672 ms (10.8 tok/s)   decode: 24 tokens in 3062 ms (7.84 tok/s)
memory pressure during run: 8025 decompressions/s, 0 swap-ins/s, compressed +0.62 GiB, swap +0.00 GiB
experts: 15992 loads, hit rate 56.7%, 9.7 MiB read per acquire-set, I/O wait 2.46 ms per segment avg, segment overhead 2.47 ms avg over 1225 segments
--- pool 2.5 GiB ---
prompt: 18 tokens in 1593 ms (11.3 tok/s)   decode: 24 tokens in 2473 ms (9.71 tok/s)
experts: 15992 loads, hit rate 71.6%, 6.3 MiB read per acquire-set, I/O wait 2.13 ms per segment avg, segment overhead 2.15 ms avg over 1225 segments
```
Result: top logits identical across pool sizes (25 rows): PASS

Generated text (pool 1 GiB):
```
Explain in detail how a hash map handles collisions, and compare chaining with open addressing. Also
,
 explain
 the
 difference
```

## Routing trace (200 sampled tokens) and cache simulation
```
experts: 83576 loads, hit rate 69.6%, 4.4 MiB read per acquire-set, I/O wait 1.28 ms per segment avg, segment overhead 1.31 ms avg over 9849 segments
layers=48 experts=128 top_k=8 tokens=218 expert loads per token=384
mean consecutive-token expert overlap per layer: 40.6%
hottest 25% of experts per layer cover 71.2% of loads
hottest 50% of experts per layer cover 94.5% of loads

| pool per layer | slots | LRU hit | LFU-decay hit | Belady ceiling | LRU misses/token | est. tok/s (LRU) |
|---|---|---|---|---|---|---|
| 12.5% | 16 | 53.8% | 48.8% | 72.9% | 177.5 | 7.2 |
| 25.0% | 32 | 76.1% | 72.3% | 86.7% | 91.9 | 12.3 |
| 37.5% | 48 | 86.5% | 85.0% | 91.9% | 52.0 | 18.2 |
| 50.0% | 64 | 91.5% | 91.2% | 94.0% | 32.5 | 23.9 |
| 62.5% | 80 | 93.7% | 93.8% | 94.5% | 24.1 | 27.6 |
| 75.0% | 96 | 94.5% | 94.5% | 94.6% | 21.2 | 29.3 |
```

## Decode speed by pool size (--bench 64 48, streaming)
```
pool 1 GiB:
memory pressure during run: 6845 decompressions/s, 2 swap-ins/s, compressed -0.04 GiB, swap +0.00 GiB
experts: 54928 loads, hit rate 73.2%, 10.1 MiB read per acquire-set, I/O wait 2.34 ms per segment avg, segment overhead 2.35 ms avg over 2499 segments
bench: pp64 47.7 tok/s (1341 ms)   tg48 8.21 tok/s (121.8 ms/token)
pool 2 GiB:
memory pressure during run: 5823 decompressions/s, 0 swap-ins/s, compressed +0.22 GiB, swap +0.00 GiB
experts: 54928 loads, hit rate 80.8%, 7.3 MiB read per acquire-set, I/O wait 2.04 ms per segment avg, segment overhead 2.05 ms avg over 2499 segments
bench: pp64 39.1 tok/s (1637 ms)   tg48 10.85 tok/s (92.2 ms/token)
pool 3 GiB:
pool 4 GiB:
```

Resident run and llama.cpp baseline skipped: only 5.0 GiB available (need about 13 GiB free for this file).

## Rerun of the 3 and 4 GiB pools after the pool-shard fix (2026-10-11 21:14)

The empty 3 and 4 GiB rows above were a crash: ggml's Metal `mul_mat_id` prefill kernel asserts
that an expert tensor holds at most 1024 experts (one thread per expert in its id-mapping
kernel), and a 3 GiB pool put more than 1024 slots into one pool tensor. Pools are now split
into shards of at most 1024 slots, each covering a contiguous range of layers.

```
pool 3 GiB:
memory pressure during run: 7584 decompressions/s, 0 swap-ins/s, compressed +0.56 GiB, swap +0.00 GiB
experts: 54928 loads, hit rate 84.6%, 5.8 MiB read per acquire-set, I/O wait 1.87 ms per segment avg, segment overhead 1.89 ms avg over 2499 segments
bench: pp64 38.5 tok/s (1663 ms)   tg48 12.22 tok/s (81.8 ms/token)
pool 4 GiB:
memory pressure during run: 9512 decompressions/s, 6 swap-ins/s, compressed +0.13 GiB, swap +0.00 GiB
experts: 54928 loads, hit rate 87.7%, 4.7 MiB read per acquire-set, I/O wait 1.51 ms per segment avg, segment overhead 1.53 ms avg over 2499 segments
bench: pp64 50.1 tok/s (1278 ms)   tg48 14.55 tok/s (68.7 ms/token)
```

Pool layout at 4 GiB (`--verbose`):
```
  pool q2_K/q2_K/q3_K: 19 layers, 995 slots x 1.63 MiB = 1.58 GiB
  pool q2_K/q2_K/q3_K: 18 layers, 942 slots x 1.63 MiB = 1.50 GiB
  pool q2_K/q2_K/q4_K: 11 layers, 513 slots x 1.83 MiB = 0.92 GiB
```
