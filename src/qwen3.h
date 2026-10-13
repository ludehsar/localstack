// qwen3.h - the model definition for the Qwen2 / Qwen3 / Qwen3-MoE family.
//
// Reads hyperparameters from GGUF metadata, binds weight tensors by name, and builds ggml compute
// graphs for one forward pass. A graph is a description of tensor operations; nothing runs until
// the Engine hands it to the scheduler. Each layer of the transformer is:
//
//   x = x + Attention(RMSNorm(x))      // mixes information across tokens, reads/writes the KV cache
//   x = x + FFN(RMSNorm(x))            // per-token MLP: dense SwiGLU, or MoE (router picks top-k experts)
//
// followed at the end by a final RMSNorm and the output projection to vocabulary logits.

#pragma once

#include "gguf_model.h"
#include "kv_cache.h"
#include "expert_store.h"

#include <string>
#include <vector>

// Model shape, read from the "<arch>.*" GGUF keys.
struct Qwen3HParams {
    uint32_t n_layer = 0;
    // Width of the residual stream (hidden size).
    uint32_t n_embd = 0;
    // Hidden size of the dense FFN.
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    // Grouped-query attention: n_head query heads share n_head_kv key/value heads, shrinking the KV cache.
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ctx_train = 0;
    float rms_eps = 1e-6f;
    // Base frequency of the rotary position embedding (RoPE).
    float rope_freq_base = 10000.0f;
    // MoE: experts per layer, experts used per token (top-k), and each expert's FFN width. 0 = dense model.
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_ff_exp = 0;

    bool moe() const { return n_expert > 0; }
    // Width of one token's K (or V) row in the cache: all KV heads concatenated.
    uint32_t n_embd_k_gqa() const { return head_dim * n_head_kv; }
    uint32_t n_embd_v_gqa() const { return head_dim * n_head_kv; }
};

// One transformer block's weights. Unused members stay null depending on the variant.
struct Qwen3Layer {
    ggml_tensor *attn_norm = nullptr;
    ggml_tensor *wq = nullptr;
    ggml_tensor *wk = nullptr;
    ggml_tensor *wv = nullptr;
    ggml_tensor *wo = nullptr;
    // Q/K/V biases: present in Qwen2 only.
    ggml_tensor *bq = nullptr;
    ggml_tensor *bk = nullptr;
    ggml_tensor *bv = nullptr;
    // Per-head RMSNorm on Q and K before RoPE: Qwen3 only.
    ggml_tensor *q_norm = nullptr;
    ggml_tensor *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr;
    // Dense FFN (SwiGLU): down(silu(gate(x)) * up(x)).
    ggml_tensor *ffn_gate = nullptr;
    ggml_tensor *ffn_up = nullptr;
    ggml_tensor *ffn_down = nullptr;
    // MoE router: scores every expert for each token.
    ggml_tensor *ffn_gate_inp = nullptr;
    // MoE experts stacked along dim 2 ([.., .., n_expert]). Null in streaming mode; the ExpertStore pools stand in.
    ggml_tensor *ffn_up_exps = nullptr;
    ggml_tensor *ffn_gate_exps = nullptr;
    ggml_tensor *ffn_down_exps = nullptr;
};

// All model weights. `output` aliases tok_embd when the model ties input and output embeddings.
struct Qwen3Weights {
    ggml_tensor *tok_embd = nullptr;
    ggml_tensor *output_norm = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<Qwen3Layer> layers;
};

// Per-step graph inputs, filled by Engine::make_step_inputs.
struct Qwen3StepInputs {
    // [n_tokens] token ids.
    ggml_tensor *tokens = nullptr;
    // [n_tokens] positions within the sequence, for RoPE.
    ggml_tensor *pos = nullptr;
    // [n_tokens] KV cache cell each new token's K/V is written to.
    ggml_tensor *kv_idxs = nullptr;
    // [n_kv, n_tokens] attention mask from KvCache::fill_mask.
    ggml_tensor *mask = nullptr;
    // [1] which row to compute logits for (the last token).
    ggml_tensor *out_ids = nullptr;
};

// Streaming: GPU tensors that outlive a graph segment, carrying the residual stream, the FFN-normed
// hidden state, and the router weights from "attention of layer l" to "experts of layer l".
struct Qwen3Persist {
    ggml_tensor *resid = nullptr;
    ggml_tensor *hnorm = nullptr;
    ggml_tensor *weights = nullptr;
};

// Streaming: the expert-pool tensors (indexed by slot) a segment's MoE step reads from.
struct Qwen3PoolTensors {
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *down = nullptr;
};

// True for the stacked expert tensors ("*_exps.weight"), which streaming mode leaves on disk.
bool qwen3_is_expert_tensor(const std::string &name);
// Reads and validates hyperparameters; rejects architectures other than qwen2/qwen3/qwen3moe.
bool qwen3_load_hparams(const GgufModel &m, Qwen3HParams &hp, std::string &err);
// Binds loaded tensors to Qwen3Weights by GGUF name; also sets hp.n_vocab.
bool qwen3_map_weights(const GgufModel &m, Qwen3HParams &hp, Qwen3Weights &w, bool stream_experts, std::string &err);
// Collects file offsets/shapes of every layer's expert tensors for the ExpertStore.
bool qwen3_expert_infos(const GgufModel &m, const Qwen3HParams &hp, std::vector<ExpertLayerInfo> &infos, std::string &err);

// Resident path: the whole model in one graph. `selected_out` gets each MoE layer's chosen expert ids
// (for routing traces); `logits_out` the [n_vocab] logits of the last token.
ggml_cgraph *qwen3_build_graph(ggml_context *ctx, size_t graph_size, const Qwen3HParams &hp, const Qwen3Weights &w,
                               const KvCache &kv, const Qwen3StepInputs &in, int n_tokens, int n_kv,
                               std::vector<ggml_tensor *> &selected_out, ggml_tensor **logits_out);

// Streaming path: one segment = experts of `moe_layer` (via pool slots in `slot_ids_out`, filled by the
// CPU before compute), then attention + router of `attn_layer`, whose chosen experts go to `selected_out`.
// moe_layer = -1 starts from the token embeddings; attn_layer = -1 ends with the output head.
ggml_cgraph *qwen3_build_segment(ggml_context *ctx, size_t graph_size, const Qwen3HParams &hp, const Qwen3Weights &w,
                                 const KvCache &kv, const Qwen3StepInputs &in, const Qwen3Persist &p, int n_tokens, int n_kv,
                                 int moe_layer, const Qwen3PoolTensors *pool, ggml_tensor **slot_ids_out, int attn_layer,
                                 ggml_tensor **selected_out, ggml_tensor **logits_out);
