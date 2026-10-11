#include "kv_cache.h"

#include <cmath>

KvCache::~KvCache() {
    free();
}

bool KvCache::init(ggml_backend_t backend, uint32_t n_layer, uint32_t n_embd_k_gqa, uint32_t n_embd_v_gqa, uint32_t sz,
                   ggml_type t, std::string &err) {
    free();
    size = sz;
    type = t;
    ggml_init_params params = { ggml_tensor_overhead() * 2 * n_layer + 4096, nullptr, true };
    ctx = ggml_init(params);
    k.resize(n_layer);
    v.resize(n_layer);
    for (uint32_t l = 0; l < n_layer; l++) {
        k[l] = ggml_new_tensor_2d(ctx, type, n_embd_k_gqa, size);
        v[l] = ggml_new_tensor_2d(ctx, type, n_embd_v_gqa, size);
        ggml_format_name(k[l], "cache_k_l%u", l);
        ggml_format_name(v[l], "cache_v_l%u", l);
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        err = "failed to allocate the KV cache";
        return false;
    }
    ggml_backend_buffer_clear(buffer, 0);

    cell_seq.assign(size, -1);
    cell_pos.assign(size, -1);
    free_blocks.clear();
    for (uint32_t b = size / block_size; b > 0; b--) free_blocks.push_back(b - 1);
    seq_blocks.clear();
    seq_len.clear();
    used_hwm = 0;
    return true;
}

void KvCache::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    buffer = nullptr;
    ctx = nullptr;
    k.clear();
    v.clear();
}

size_t KvCache::bytes() const {
    return buffer ? ggml_backend_buffer_get_size(buffer) : 0;
}

bool KvCache::alloc(int32_t seq, int n_tokens, std::vector<int64_t> &cells, std::vector<int32_t> &positions) {
    int32_t &len = seq_len[seq];
    std::vector<uint32_t> &blocks = seq_blocks[seq];
    const uint32_t have = (uint32_t) blocks.size() * block_size;
    const uint32_t need = (uint32_t) len + (uint32_t) n_tokens;
    const uint32_t extra_blocks = need > have ? (need - have + block_size - 1) / block_size : 0;
    if (extra_blocks > free_blocks.size()) return false;
    for (uint32_t i = 0; i < extra_blocks; i++) {
        blocks.push_back(free_blocks.back());
        free_blocks.pop_back();
    }
    cells.resize(n_tokens);
    positions.resize(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        const uint32_t cell = blocks[(uint32_t) len / block_size] * block_size + (uint32_t) len % block_size;
        cell_seq[cell] = seq;
        cell_pos[cell] = len;
        cells[i] = cell;
        positions[i] = len;
        if (cell + 1 > used_hwm) used_hwm = cell + 1;
        len++;
    }
    return true;
}

void KvCache::remove_seq(int32_t seq) {
    auto it = seq_blocks.find(seq);
    if (it == seq_blocks.end()) return;
    for (uint32_t b : it->second) {
        for (uint32_t c = b * block_size; c < (b + 1) * block_size; c++) {
            cell_seq[c] = -1;
            cell_pos[c] = -1;
        }
        free_blocks.push_back(b);
    }
    seq_blocks.erase(it);
    seq_len.erase(seq);
    used_hwm = 0;
    for (uint32_t c = 0; c < size; c++) {
        if (cell_seq[c] >= 0) used_hwm = c + 1;
    }
}

void KvCache::truncate_seq(int32_t seq, int32_t keep) {
    auto it = seq_blocks.find(seq);
    if (it == seq_blocks.end()) return;
    if (keep <= 0) {
        remove_seq(seq);
        return;
    }
    std::vector<uint32_t> &blocks = it->second;
    int32_t &len = seq_len[seq];
    if (keep >= len) return;
    for (int32_t pos = keep; pos < len; pos++) {
        const uint32_t cell = blocks[(uint32_t) pos / block_size] * block_size + (uint32_t) pos % block_size;
        cell_seq[cell] = -1;
        cell_pos[cell] = -1;
    }
    const uint32_t keep_blocks = ((uint32_t) keep + block_size - 1) / block_size;
    while (blocks.size() > keep_blocks) {
        free_blocks.push_back(blocks.back());
        blocks.pop_back();
    }
    len = keep;
    used_hwm = 0;
    for (uint32_t c = 0; c < size; c++) {
        if (cell_seq[c] >= 0) used_hwm = c + 1;
    }
}

int32_t KvCache::length(int32_t seq) const {
    auto it = seq_len.find(seq);
    return it == seq_len.end() ? 0 : it->second;
}

void KvCache::fill_mask(int32_t seq, const int32_t *pos, int n_tokens, int n_kv, ggml_fp16_t *mask) const {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_tokens; i++) {
        ggml_fp16_t *row = mask + (size_t) i * n_kv;
        for (int j = 0; j < n_kv; j++) {
            const bool visible = cell_seq[j] == seq && cell_pos[j] <= pos[i];
            row[j] = visible ? zero : neg_inf;
        }
    }
}
