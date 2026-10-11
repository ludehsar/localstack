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

struct ExpertLayerInfo {
    GgufTensorInfo up;
    GgufTensorInfo gate;
    GgufTensorInfo down;
};

struct IoTask {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
};

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
    int pending_ = 0;
    bool stop_ = false;
};

struct ExpertStoreStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_read = 0;
    double wait_ms = 0.0;
    uint64_t acquires = 0;
};

struct ExpertPool {
    ggml_type tu, tg, td;
    size_t up_bytes, gate_bytes, down_bytes;
    uint32_t n_slots = 0;
    std::vector<int> layers;
    void *mem_up = nullptr;
    void *mem_gate = nullptr;
    void *mem_down = nullptr;
    size_t bytes_up = 0, bytes_gate = 0, bytes_down = 0;
    ggml_backend_buffer_t buf_up = nullptr, buf_gate = nullptr, buf_down = nullptr;
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *down = nullptr;
    std::vector<int32_t> owner_layer;
    std::vector<int32_t> owner_expert;
    std::vector<int32_t> refcount;
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t lru_head = -1;
    int32_t lru_tail = -1;
    uint32_t n_in_lru = 0;
};

class ExpertStore {
public:
    ~ExpertStore();
    bool init(ggml_backend_dev_t dev, ggml_backend_t backend, int fd, uint32_t n_layer, uint32_t n_expert, int64_t n_embd,
              int64_t n_ff_exp, const std::vector<ExpertLayerInfo> &layers, size_t pool_bytes, uint32_t min_slots, int io_threads,
              std::string &err);
    const ExpertPool &pool_for(int layer) const { return pools_[layer_pool_[layer]]; }
    bool acquire(int layer, const int32_t *experts, int n, int32_t *slots, std::string &err);
    void release_all();
    const ExpertStoreStats &stats() const { return stats_; }
    size_t total_bytes() const;
    uint32_t total_slots() const;
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
    std::vector<int32_t> slot_of_;
    std::vector<std::pair<int, int32_t>> pinned_;
    IoPool io_;
    ExpertStoreStats stats_;
};
