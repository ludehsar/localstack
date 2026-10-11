#pragma once

#include "gguf_model.h"
#include "kv_cache.h"
#include "expert_store.h"

#include <string>
#include <vector>

struct Qwen3HParams {
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ctx_train = 0;
    float rms_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_ff_exp = 0;

    bool moe() const { return n_expert > 0; }
    uint32_t n_embd_k_gqa() const { return head_dim * n_head_kv; }
    uint32_t n_embd_v_gqa() const { return head_dim * n_head_kv; }
};

struct Qwen3Layer {
    ggml_tensor *attn_norm = nullptr;
    ggml_tensor *wq = nullptr;
    ggml_tensor *wk = nullptr;
    ggml_tensor *wv = nullptr;
    ggml_tensor *wo = nullptr;
    ggml_tensor *bq = nullptr;
    ggml_tensor *bk = nullptr;
    ggml_tensor *bv = nullptr;
    ggml_tensor *q_norm = nullptr;
    ggml_tensor *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr;
    ggml_tensor *ffn_gate = nullptr;
    ggml_tensor *ffn_up = nullptr;
    ggml_tensor *ffn_down = nullptr;
    ggml_tensor *ffn_gate_inp = nullptr;
    ggml_tensor *ffn_up_exps = nullptr;
    ggml_tensor *ffn_gate_exps = nullptr;
    ggml_tensor *ffn_down_exps = nullptr;
};

struct Qwen3Weights {
    ggml_tensor *tok_embd = nullptr;
    ggml_tensor *output_norm = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<Qwen3Layer> layers;
};

struct Qwen3StepInputs {
    ggml_tensor *tokens = nullptr;
    ggml_tensor *pos = nullptr;
    ggml_tensor *kv_idxs = nullptr;
    ggml_tensor *mask = nullptr;
    ggml_tensor *out_ids = nullptr;
};

struct Qwen3Persist {
    ggml_tensor *resid = nullptr;
    ggml_tensor *hnorm = nullptr;
    ggml_tensor *weights = nullptr;
};

struct Qwen3PoolTensors {
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *down = nullptr;
};

bool qwen3_is_expert_tensor(const std::string &name);
bool qwen3_load_hparams(const GgufModel &m, Qwen3HParams &hp, std::string &err);
bool qwen3_map_weights(const GgufModel &m, Qwen3HParams &hp, Qwen3Weights &w, bool stream_experts, std::string &err);
bool qwen3_expert_infos(const GgufModel &m, const Qwen3HParams &hp, std::vector<ExpertLayerInfo> &infos, std::string &err);

ggml_cgraph *qwen3_build_graph(ggml_context *ctx, size_t graph_size, const Qwen3HParams &hp, const Qwen3Weights &w,
                               const KvCache &kv, const Qwen3StepInputs &in, int n_tokens, int n_kv,
                               std::vector<ggml_tensor *> &selected_out, ggml_tensor **logits_out);

ggml_cgraph *qwen3_build_segment(ggml_context *ctx, size_t graph_size, const Qwen3HParams &hp, const Qwen3Weights &w,
                                 const KvCache &kv, const Qwen3StepInputs &in, const Qwen3Persist &p, int n_tokens, int n_kv,
                                 int moe_layer, const Qwen3PoolTensors *pool, ggml_tensor **slot_ids_out, int attn_layer,
                                 ggml_tensor **selected_out, ggml_tensor **logits_out);
