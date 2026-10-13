// kv_cache.h - the attention key/value cache.
//
// Every token the model has seen leaves a key and a value vector in every layer; attention for a
// new token reads all of them. Caching them means each token is processed once, not once per
// generated token. Storage is one [n_embd_gqa, size] tensor per layer for K and one for V; a column
// is a "cell" holding one token.
//
// Cells are handed out in 32-token blocks, like the pages of vLLM's PagedAttention: a sequence owns
// a list of blocks that need not be contiguous, and each cell records which sequence and position it
// holds. Attention runs over a dense prefix of cells, and fill_mask() hides cells that belong to
// other sequences or to future positions. See docs/decisions.md for why this layout was chosen.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Plain struct; Engine owns one and Qwen3 graph builders read k/v directly.
struct KvCache {
    // Total cells (tokens) and cells per block.
    uint32_t size = 0;
    uint32_t block_size = 32;
    ggml_type type = GGML_TYPE_F16;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    // Per-layer K and V tensors, [n_embd_gqa, size], in fp16.
    std::vector<ggml_tensor *> k;
    std::vector<ggml_tensor *> v;

    // Per cell: owning sequence id and token position, or -1 when free.
    std::vector<int32_t> cell_seq;
    std::vector<int32_t> cell_pos;
    // Unused block ids (a stack; init orders it so low blocks are handed out first).
    std::vector<uint32_t> free_blocks;
    // Per sequence: its blocks in position order (the block table) and its length in tokens.
    std::unordered_map<int32_t, std::vector<uint32_t>> seq_blocks;
    std::unordered_map<int32_t, int32_t> seq_len;
    // High-water mark: one past the highest occupied cell; attention never scans beyond it.
    uint32_t used_hwm = 0;

    ~KvCache();
    bool init(ggml_backend_t backend, uint32_t n_layer, uint32_t n_embd_k_gqa, uint32_t n_embd_v_gqa, uint32_t size,
              ggml_type type, std::string &err);
    void free();
    size_t bytes() const;

    // Reserves cells for the next n_tokens of `seq`, returning their cell indices and positions.
    // Returns false (allocating nothing) if there are not enough free blocks.
    bool alloc(int32_t seq, int n_tokens, std::vector<int64_t> &cells, std::vector<int32_t> &positions);
    void remove_seq(int32_t seq);
    void truncate_seq(int32_t seq, int32_t keep);
    int32_t length(int32_t seq) const;
    uint32_t n_free_cells() const { return (uint32_t) free_blocks.size() * block_size; }
    // Builds the [n_kv x n_tokens] attention mask: 0 where token i may attend to cell j (same sequence,
    // position <= its own), -inf elsewhere. This one mask gives both causality and sequence isolation.
    void fill_mask(int32_t seq, const int32_t *pos, int n_tokens, int n_kv, ggml_fp16_t *mask) const;
};
