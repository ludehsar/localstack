// expert_store.h - the SSD-backed cache of MoE expert weights (the core of streaming mode).
//
// A Qwen3-30B-A3B layer has 128 experts but each token uses only 8, so most expert weights are
// idle at any moment. Instead of holding all of them in RAM, the ExpertStore keeps a fixed number
// of "slots" per pool, each big enough for one expert's three matrices (up, gate, down), and reads
// experts from the GGUF file into slots on demand. The GPU sees each pool as three 3-D tensors
// [.., .., n_slots], so the graph indexes experts by slot number instead of expert number.
//
// Pools: layers whose experts share the same quantization types share a pool (a tensor has one
// type), and a pool is split into shards of at most 1024 slots because ggml's Metal mul_mat_id
// kernel cannot address more. Each layer maps to exactly one pool, so lookups stay O(1).
//
// Replacement: slots in use by the current graph segment are pinned (refcount > 0); unpinned
// slots sit in a per-pool LRU list and the least recently used one is evicted on a miss.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf_model.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Where one layer's three stacked expert tensors live in the GGUF file.
struct ExpertLayerInfo {
    GgufTensorInfo up;
    GgufTensorInfo gate;
    GgufTensorInfo down;
};

// One positioned read: copy `size` bytes at file `offset` into `dst`.
struct IoTask {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
};

// A tiny thread pool that runs pread() calls in parallel. NVMe SSDs only reach full bandwidth with
// several reads in flight (see tools/ssd_bench.c), so a cache miss's 3 reads, and the reads of all
// misses in one acquire(), are queued here and awaited together with wait_all().
class IoPool {
public:
    void start(int n_threads);
    void stop();
    void submit(const IoTask &task);
    void wait_all();
    ~IoPool() { stop(); }

private:
    void worker();
    std::vector<std::thread> threads_;
    std::deque<IoTask> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable done_;
    // Tasks submitted but not yet finished; wait_all() blocks until this is 0.
    int pending_ = 0;
    bool stop_ = false;
};

// Cache counters (cumulative since load).
struct ExpertStoreStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_read = 0;
    // Time spent inside acquire(), mostly waiting on SSD reads.
    double wait_ms = 0.0;
    uint64_t acquires = 0;
};

// Metal's mul_mat_id kernel asserts ne02 <= 1024 (experts per weight tensor), hence the shard size.
constexpr uint32_t kMaxSlotsPerPool = 1024;

// One pool shard: slot storage for a set of layers that share expert quant types.
struct ExpertPool {
    // Quant types and per-expert byte sizes of the up / gate / down matrices.
    ggml_type tu, tg, td;
    size_t up_bytes, gate_bytes, down_bytes;
    uint32_t n_slots = 0;
    // Layers served by this pool; each layer looks only in its own pool.
    std::vector<int> layers;
    // Page-aligned host memory for each matrix, wrapped zero-copy as a Metal buffer (unified memory).
    void *mem_up = nullptr;
    void *mem_gate = nullptr;
    void *mem_down = nullptr;
    size_t bytes_up = 0, bytes_gate = 0, bytes_down = 0;
    ggml_backend_buffer_t buf_up = nullptr, buf_gate = nullptr, buf_down = nullptr;
    // The [n_embd, n_ff_exp, n_slots]-shaped tensors the graph reads expert weights from.
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *down = nullptr;
    // Per slot: which (layer, expert) it currently holds (-1 = empty) and how many users pin it.
    std::vector<int32_t> owner_layer;
    std::vector<int32_t> owner_expert;
    std::vector<int32_t> refcount;
    // Intrusive doubly linked LRU list of unpinned slots, stored as index arrays. Head = evict next.
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t lru_head = -1;
    int32_t lru_tail = -1;
    uint32_t n_in_lru = 0;
};

// Owns all pools, maps (layer, expert) -> slot, and loads misses from the GGUF file descriptor.
class ExpertStore {
public:
    ~ExpertStore();
    // Splits `pool_bytes` across layers, allocates pools (each at least `min_slots` slots), starts I/O threads.
    bool init(ggml_backend_dev_t dev, ggml_backend_t backend, int fd, uint32_t n_layer, uint32_t n_expert, int64_t n_embd,
              int64_t n_ff_exp, const std::vector<ExpertLayerInfo> &layers, size_t pool_bytes, uint32_t min_slots, int io_threads,
              std::string &err);
    const ExpertPool &pool_for(int layer) const { return pools_[layer_pool_[layer]]; }
    // Makes `n` experts of `layer` resident, writing their slot numbers to `slots` and pinning them.
    // Blocks until any needed SSD reads finish. Fails if the layer needs more slots than its pool has.
    bool acquire(int layer, const int32_t *experts, int n, int32_t *slots, std::string &err);
    // Unpins everything acquired since the last release (called after each graph segment runs).
    void release_all();
    const ExpertStoreStats &stats() const { return stats_; }
    size_t total_bytes() const;
    uint32_t total_slots() const;
    // Human-readable pool layout for --verbose.
    std::string describe() const;

private:
    void lru_push_tail(ExpertPool &p, int32_t slot);
    void lru_unlink(ExpertPool &p, int32_t slot);
    int32_t lru_pop_head(ExpertPool &p);

    int fd_ = -1;
    uint32_t n_expert_ = 0;
    ggml_context *ctx_ = nullptr;
    std::vector<ExpertPool> pools_;
    std::vector<int> layer_pool_;
    std::vector<ExpertLayerInfo> infos_;
    // slot_of_[layer * n_expert + expert] = slot holding that expert, or -1 if not resident.
    std::vector<int32_t> slot_of_;
    // (pool index, slot) pairs pinned by acquire() and not yet released.
    std::vector<std::pair<int, int32_t>> pinned_;
    IoPool io_;
    ExpertStoreStats stats_;
};
