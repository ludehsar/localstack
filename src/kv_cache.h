#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct KvCache {
    uint32_t size = 0;
    uint32_t block_size = 32;
    ggml_type type = GGML_TYPE_F16;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> k;
    std::vector<ggml_tensor *> v;

    std::vector<int32_t> cell_seq;
    std::vector<int32_t> cell_pos;
    std::vector<uint32_t> free_blocks;
    std::unordered_map<int32_t, std::vector<uint32_t>> seq_blocks;
    std::unordered_map<int32_t, int32_t> seq_len;
    uint32_t used_hwm = 0;

    ~KvCache();
    bool init(ggml_backend_t backend, uint32_t n_layer, uint32_t n_embd_k_gqa, uint32_t n_embd_v_gqa, uint32_t size,
              ggml_type type, std::string &err);
    void free();
    size_t bytes() const;

    bool alloc(int32_t seq, int n_tokens, std::vector<int64_t> &cells, std::vector<int32_t> &positions);
    void remove_seq(int32_t seq);
    void truncate_seq(int32_t seq, int32_t keep);
    int32_t length(int32_t seq) const;
    uint32_t n_free_cells() const { return (uint32_t) free_blocks.size() * block_size; }
    void fill_mask(int32_t seq, const int32_t *pos, int n_tokens, int n_kv, ggml_fp16_t *mask) const;
};
