#!/usr/bin/env python3
import argparse
import collections
import struct
import sys


def read_trace(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"LSTKRTE1":
            sys.exit("not a routing trace")
        n_layer, n_expert, k = struct.unpack("<iii", f.read(12))
        per_layer = [[] for _ in range(n_layer)]
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            n_tokens = struct.unpack("<i", head)[0]
            step = []
            for l in range(n_layer):
                raw = f.read(4 * k * n_tokens)
                if len(raw) < 4 * k * n_tokens:
                    step = None
                    break
                step.append(struct.unpack("<%di" % (k * n_tokens), raw))
            if step is None:
                break
            for l, ids in enumerate(step):
                for t in range(n_tokens):
                    row = ids[t * k:(t + 1) * k]
                    if row[0] >= 0:
                        per_layer[l].append(tuple(row))
    return n_layer, n_expert, k, per_layer


def sim_lru(seq, cap):
    cache = collections.OrderedDict()
    hits = 0
    for experts in seq:
        for e in experts:
            if e in cache:
                cache.move_to_end(e)
                hits += 1
            else:
                if len(cache) >= cap:
                    cache.popitem(last=False)
                cache[e] = True
    return hits


def sim_lfu_decay(seq, cap, decay=0.98):
    score = {}
    cache = set()
    hits = 0
    for experts in seq:
        for e in list(score):
            score[e] *= decay
        for e in experts:
            score[e] = score.get(e, 0.0) + 1.0
            if e in cache:
                hits += 1
            else:
                if len(cache) >= cap:
                    victim = min(cache, key=lambda x: score.get(x, 0.0))
                    cache.remove(victim)
                cache.add(e)
    return hits


def sim_belady(seq, cap):
    flat = [e for experts in seq for e in experts]
    next_use = [None] * len(flat)
    last = {}
    for i in range(len(flat) - 1, -1, -1):
        next_use[i] = last.get(flat[i], float("inf"))
        last[flat[i]] = i
    cache = {}
    hits = 0
    for i, e in enumerate(flat):
        if e in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                victim = max(cache, key=lambda x: cache[x])
                del cache[victim]
        cache[e] = next_use[i]
    return hits


def main():
    ap = argparse.ArgumentParser(description="replay a routing trace against expert cache policies")
    ap.add_argument("trace")
    ap.add_argument("--fractions", default="0.125,0.25,0.375,0.5,0.625,0.75")
    ap.add_argument("--miss-ms", type=float, default=0.67, help="SSD cost of one expert miss in ms")
    ap.add_argument("--compute-ms", type=float, default=20.0, help="non-I/O cost per token in ms")
    args = ap.parse_args()

    n_layer, n_expert, k, per_layer = read_trace(args.trace)
    n_tok = max(len(s) for s in per_layer)
    loads_per_token = n_layer * k
    print("layers=%d experts=%d top_k=%d tokens=%d expert loads per token=%d" % (n_layer, n_expert, k, n_tok, loads_per_token))

    overlap = 0.0
    counts = collections.Counter()
    for seq in per_layer:
        for i in range(1, len(seq)):
            overlap += len(set(seq[i]) & set(seq[i - 1])) / k
        for experts in seq:
            counts.update(experts)
    print("mean consecutive-token expert overlap per layer: %.1f%%" % (100.0 * overlap / max(1, n_layer * (n_tok - 1))))
    total = sum(counts.values())
    per_layer_sorted = []
    for seq in per_layer:
        c = collections.Counter(e for experts in seq for e in experts)
        per_layer_sorted.append(sorted(c.values(), reverse=True))
    for frac in (0.25, 0.5):
        covered = 0
        for vals in per_layer_sorted:
            covered += sum(vals[:int(n_expert * frac)])
        print("hottest %.0f%% of experts per layer cover %.1f%% of loads" % (100 * frac, 100.0 * covered / total))

    fractions = [float(x) for x in args.fractions.split(",")]
    print()
    print("| pool per layer | slots | LRU hit | LFU-decay hit | Belady ceiling | LRU misses/token | est. tok/s (LRU) |")
    print("|---|---|---|---|---|---|---|")
    for frac in fractions:
        cap = max(k, int(n_expert * frac))
        lru = sum(sim_lru(seq, cap) for seq in per_layer)
        lfu = sum(sim_lfu_decay(seq, cap) for seq in per_layer)
        bel = sum(sim_belady(seq, cap) for seq in per_layer)
        misses = (total - lru) / n_tok
        est = 1000.0 / (args.compute_ms + misses * args.miss_ms)
        print("| %.1f%% | %d | %.1f%% | %.1f%% | %.1f%% | %.1f | %.1f |" % (100 * frac, cap, 100.0 * lru / total, 100.0 * lfu / total,
                                                                   100.0 * bel / total, misses, est))


if __name__ == "__main__":
    main()
