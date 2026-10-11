#include "expert_store.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

void IoPool::start(int n_threads) {
    stop_ = false;
    for (int i = 0; i < n_threads; i++) threads_.emplace_back([this] { worker(); });
}

void IoPool::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread &t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

void IoPool::submit(const IoTask &task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(task);
        pending_++;
    }
    cv_.notify_one();
}

void IoPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
}

void IoPool::worker() {
    for (;;) {
        IoTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            task = queue_.front();
            queue_.pop_front();
        }
        uint8_t *dst = (uint8_t *) task.dst;
        size_t left = task.size;
        off_t off = task.offset;
        while (left > 0) {
            ssize_t n = pread(task.fd, dst, left, off);
            if (n <= 0) {
                fprintf(stderr, "expert read failed at offset %lld\n", (long long) off);
                break;
            }
            dst += n;
            left -= (size_t) n;
            off += n;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_--;
            if (pending_ == 0) done_.notify_all();
        }
    }
}

ExpertStore::~ExpertStore() {
    io_.stop();
    for (ExpertPool &p : pools_) {
        if (p.buf_up) ggml_backend_buffer_free(p.buf_up);
        if (p.buf_gate) ggml_backend_buffer_free(p.buf_gate);
        if (p.buf_down) ggml_backend_buffer_free(p.buf_down);
        if (p.mem_up) free(p.mem_up);
        if (p.mem_gate) free(p.mem_gate);
        if (p.mem_down) free(p.mem_down);
    }
    if (ctx_) ggml_free(ctx_);
}

static void *alloc_pages(size_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 2u << 20, bytes) != 0) return nullptr;
    return p;
}

static size_t page_round(size_t bytes) {
    const size_t page = 16384;
    return (bytes + page - 1) / page * page;
}

bool ExpertStore::init(ggml_backend_dev_t dev, ggml_backend_t backend, int fd, uint32_t n_layer, uint32_t n_expert,
                       int64_t n_embd, int64_t n_ff_exp, const std::vector<ExpertLayerInfo> &layers, size_t pool_bytes,
                       uint32_t min_slots, int io_threads, std::string &err) {
    (void) backend;
    fd_ = fd;
    n_expert_ = n_expert;
    infos_ = layers;
    layer_pool_.assign(n_layer, -1);
    slot_of_.assign((size_t) n_layer * n_expert, -1);

    std::map<std::tuple<int, int, int>, int> classes;
    for (uint32_t l = 0; l < n_layer; l++) {
        const ExpertLayerInfo &li = layers[l];
        auto key = std::make_tuple((int) li.up.type, (int) li.gate.type, (int) li.down.type);
        auto it = classes.find(key);
        if (it == classes.end()) {
            ExpertPool p;
            p.tu = li.up.type;
            p.tg = li.gate.type;
            p.td = li.down.type;
            p.up_bytes = li.up.expert_stride();
            p.gate_bytes = li.gate.expert_stride();
            p.down_bytes = li.down.expert_stride();
            pools_.push_back(p);
            it = classes.emplace(key, (int) pools_.size() - 1).first;
        }
        pools_[it->second].layers.push_back((int) l);
        layer_pool_[l] = it->second;
    }

    ggml_init_params gp = { ggml_tensor_overhead() * 3 * pools_.size() + 1024, nullptr, true };
    ctx_ = ggml_init(gp);

    for (ExpertPool &p : pools_) {
        const size_t slab = p.up_bytes + p.gate_bytes + p.down_bytes;
        const size_t share = pool_bytes * p.layers.size() / n_layer;
        p.n_slots = (uint32_t) std::max<size_t>(min_slots, share / slab);
        p.bytes_up = page_round(p.up_bytes * p.n_slots);
        p.bytes_gate = page_round(p.gate_bytes * p.n_slots);
        p.bytes_down = page_round(p.down_bytes * p.n_slots);
        p.mem_up = alloc_pages(p.bytes_up);
        p.mem_gate = alloc_pages(p.bytes_gate);
        p.mem_down = alloc_pages(p.bytes_down);
        if (!p.mem_up || !p.mem_gate || !p.mem_down) {
            err = "cannot allocate expert pool memory";
            return false;
        }
        p.buf_up = ggml_backend_dev_buffer_from_host_ptr(dev, p.mem_up, p.bytes_up, p.bytes_up);
        p.buf_gate = ggml_backend_dev_buffer_from_host_ptr(dev, p.mem_gate, p.bytes_gate, p.bytes_gate);
        p.buf_down = ggml_backend_dev_buffer_from_host_ptr(dev, p.mem_down, p.bytes_down, p.bytes_down);
        if (!p.buf_up || !p.buf_gate || !p.buf_down) {
            err = "cannot wrap expert pool memory as a GPU buffer";
            return false;
        }
        ggml_backend_buffer_set_usage(p.buf_up, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(p.buf_gate, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(p.buf_down, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        p.up = ggml_new_tensor_3d(ctx_, p.tu, n_embd, n_ff_exp, p.n_slots);
        p.gate = ggml_new_tensor_3d(ctx_, p.tg, n_embd, n_ff_exp, p.n_slots);
        p.down = ggml_new_tensor_3d(ctx_, p.td, n_ff_exp, n_embd, p.n_slots);
        if (ggml_nbytes(p.up) != p.up_bytes * p.n_slots || ggml_nbytes(p.gate) != p.gate_bytes * p.n_slots ||
            ggml_nbytes(p.down) != p.down_bytes * p.n_slots) {
            err = "expert tensor sizes in the GGUF do not match the model dimensions";
            return false;
        }
        ggml_backend_tensor_alloc(p.buf_up, p.up, p.mem_up);
        ggml_backend_tensor_alloc(p.buf_gate, p.gate, p.mem_gate);
        ggml_backend_tensor_alloc(p.buf_down, p.down, p.mem_down);
        p.owner_layer.assign(p.n_slots, -1);
        p.owner_expert.assign(p.n_slots, -1);
        p.refcount.assign(p.n_slots, 0);
        p.prev.assign(p.n_slots, -1);
        p.next.assign(p.n_slots, -1);
        for (uint32_t s = 0; s < p.n_slots; s++) lru_push_tail(p, (int32_t) s);
    }
    io_.start(std::max(1, io_threads));
    return true;
}

void ExpertStore::lru_push_tail(ExpertPool &p, int32_t slot) {
    p.prev[slot] = p.lru_tail;
    p.next[slot] = -1;
    if (p.lru_tail >= 0) p.next[p.lru_tail] = slot;
    p.lru_tail = slot;
    if (p.lru_head < 0) p.lru_head = slot;
    p.n_in_lru++;
}

void ExpertStore::lru_unlink(ExpertPool &p, int32_t slot) {
    const int32_t a = p.prev[slot];
    const int32_t b = p.next[slot];
    if (a >= 0) p.next[a] = b; else p.lru_head = b;
    if (b >= 0) p.prev[b] = a; else p.lru_tail = a;
    p.prev[slot] = -1;
    p.next[slot] = -1;
    p.n_in_lru--;
}

int32_t ExpertStore::lru_pop_head(ExpertPool &p) {
    const int32_t slot = p.lru_head;
    if (slot < 0) return -1;
    lru_unlink(p, slot);
    return slot;
}

bool ExpertStore::acquire(int layer, const int32_t *experts, int n, int32_t *slots, std::string &err) {
    ExpertPool &p = pools_[layer_pool_[layer]];
    const ExpertLayerInfo &li = infos_[layer];
    const int64_t t0 = ggml_time_us();
    int misses = 0;
    stats_.acquires++;
    for (int i = 0; i < n; i++) {
        const int32_t e = experts[i];
        int32_t &slot_ref = slot_of_[(size_t) layer * n_expert_ + e];
        if (slot_ref >= 0) {
            const int32_t s = slot_ref;
            if (p.refcount[s] == 0) lru_unlink(p, s);
            p.refcount[s]++;
            pinned_.push_back({ layer_pool_[layer], s });
            slots[i] = s;
            stats_.hits++;
            continue;
        }
        bool dup = false;
        for (int j = 0; j < i; j++) {
            if (experts[j] == e) {
                slots[i] = slots[j];
                p.refcount[slots[j]]++;
                pinned_.push_back({ layer_pool_[layer], slots[j] });
                dup = true;
                break;
            }
        }
        if (dup) continue;
        const int32_t s = lru_pop_head(p);
        if (s < 0) {
            if (misses > 0) io_.wait_all();
            err = "expert pool exhausted: layer " + std::to_string(layer) + " needs more than " + std::to_string(p.n_slots) +
                  " slots at once; raise --pool-gib or lower --n-batch";
            return false;
        }
        if (p.owner_layer[s] >= 0) slot_of_[(size_t) p.owner_layer[s] * n_expert_ + p.owner_expert[s]] = -1;
        p.owner_layer[s] = layer;
        p.owner_expert[s] = e;
        slot_ref = s;
        p.refcount[s] = 1;
        pinned_.push_back({ layer_pool_[layer], s });
        slots[i] = s;
        io_.submit({ fd_, (uint8_t *) p.mem_up + (size_t) s * p.up_bytes, (off_t) (li.up.file_offset + (size_t) e * p.up_bytes), p.up_bytes });
        io_.submit({ fd_, (uint8_t *) p.mem_gate + (size_t) s * p.gate_bytes, (off_t) (li.gate.file_offset + (size_t) e * p.gate_bytes), p.gate_bytes });
        io_.submit({ fd_, (uint8_t *) p.mem_down + (size_t) s * p.down_bytes, (off_t) (li.down.file_offset + (size_t) e * p.down_bytes), p.down_bytes });
        misses++;
        stats_.misses++;
        stats_.bytes_read += p.up_bytes + p.gate_bytes + p.down_bytes;
    }
    if (misses > 0) io_.wait_all();
    stats_.wait_ms += (ggml_time_us() - t0) / 1000.0;
    return true;
}

void ExpertStore::release_all() {
    for (const auto &[pool_idx, s] : pinned_) {
        ExpertPool &p = pools_[pool_idx];
        if (--p.refcount[s] == 0) lru_push_tail(p, s);
    }
    pinned_.clear();
}

size_t ExpertStore::total_bytes() const {
    size_t total = 0;
    for (const ExpertPool &p : pools_) total += p.bytes_up + p.bytes_gate + p.bytes_down;
    return total;
}

uint32_t ExpertStore::total_slots() const {
    uint32_t total = 0;
    for (const ExpertPool &p : pools_) total += p.n_slots;
    return total;
}

std::string ExpertStore::describe() const {
    std::string s;
    char buf[256];
    for (const ExpertPool &p : pools_) {
        snprintf(buf, sizeof(buf), "  pool %s/%s/%s: %zu layers, %u slots x %.2f MiB = %.2f GiB\n", ggml_type_name(p.tu),
                 ggml_type_name(p.tg), ggml_type_name(p.td), p.layers.size(), p.n_slots,
                 (p.up_bytes + p.gate_bytes + p.down_bytes) / 1048576.0, (p.bytes_up + p.bytes_gate + p.bytes_down) / 1073741824.0);
        s += buf;
    }
    return s;
}
