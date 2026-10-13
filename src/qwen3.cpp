// qwen3.cpp - Qwen2/Qwen3/Qwen3-MoE hyperparameters, weight binding, and graph construction.
//
// The graph builders call ggml ops that only record nodes; the Engine allocates and runs the graph.
// Tensor shapes follow ggml's convention: ne[0] is the innermost (contiguous) dimension, so a
// [n_embd, n_tokens] activation holds one token per column.

#include "qwen3.h"

#include <cmath>

bool qwen3_is_expert_tensor(const std::string &name) {
    return name.find("_exps.weight") != std::string::npos;
}

// GGUF keys are prefixed with the architecture name, e.g. "qwen3moe.block_count".
bool qwen3_load_hparams(const GgufModel &m, Qwen3HParams &hp, std::string &err) {
    if (m.arch != "qwen3" && m.arch != "qwen2" && m.arch != "qwen3moe") {
        err = "unsupported architecture: " + m.arch + " (supported: qwen2, qwen3, qwen3moe)";
        return false;
    }
    const std::string a = m.arch + ".";
    hp.n_layer = (uint32_t) m.get_int(a + "block_count");
    hp.n_embd = (uint32_t) m.get_int(a + "embedding_length");
    hp.n_ff = (uint32_t) m.get_int(a + "feed_forward_length");
    hp.n_head = (uint32_t) m.get_int(a + "attention.head_count");
    hp.n_head_kv = (uint32_t) m.get_int(a + "attention.head_count_kv", hp.n_head);
    hp.head_dim = (uint32_t) m.get_int(a + "attention.key_length", hp.n_head ? hp.n_embd / hp.n_head : 0);
    hp.n_ctx_train = (uint32_t) m.get_int(a + "context_length", 32768);
    hp.rms_eps = (float) m.get_float(a + "attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_freq_base = (float) m.get_float(a + "rope.freq_base", 10000.0);
    hp.n_expert = (uint32_t) m.get_int(a + "expert_count", 0);
    hp.n_expert_used = (uint32_t) m.get_int(a + "expert_used_count", 0);
    hp.n_ff_exp = (uint32_t) m.get_int(a + "expert_feed_forward_length", hp.n_expert_used ? hp.n_ff / hp.n_expert_used : 0);
    if (hp.moe() && (!hp.n_expert_used || !hp.n_ff_exp)) {
        err = "incomplete MoE hyperparameters in GGUF";
        return false;
    }
    if (!hp.n_layer || !hp.n_embd || !hp.n_ff || !hp.n_head || !hp.n_head_kv || !hp.head_dim) {
        err = "incomplete hyperparameters in GGUF";
        return false;
    }
    return true;
}

// Required-tensor lookup: records the first missing name in `err` so mapping can report it at the end.
static ggml_tensor *need(const GgufModel &m, const std::string &name, std::string &err) {
    ggml_tensor *t = m.tensor(name);
    if (!t && err.empty()) err = "missing tensor " + name;
    return t;
}

bool qwen3_map_weights(const GgufModel &m, Qwen3HParams &hp, Qwen3Weights &w, bool stream_experts, std::string &err) {
    w.tok_embd = need(m, "token_embd.weight", err);
    w.output_norm = need(m, "output_norm.weight", err);
    // Small models tie the output projection to the embedding matrix.
    w.output = m.tensor("output.weight");
    if (!w.output) w.output = w.tok_embd;
    w.layers.assign(hp.n_layer, Qwen3Layer{});
    for (uint32_t l = 0; l < hp.n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Qwen3Layer &L = w.layers[l];
        L.attn_norm = need(m, p + "attn_norm.weight", err);
        L.wq = need(m, p + "attn_q.weight", err);
        L.wk = need(m, p + "attn_k.weight", err);
        L.wv = need(m, p + "attn_v.weight", err);
        L.wo = need(m, p + "attn_output.weight", err);
        L.bq = m.tensor(p + "attn_q.bias");
        L.bk = m.tensor(p + "attn_k.bias");
        L.bv = m.tensor(p + "attn_v.bias");
        L.q_norm = m.tensor(p + "attn_q_norm.weight");
        L.k_norm = m.tensor(p + "attn_k_norm.weight");
        L.ffn_norm = need(m, p + "ffn_norm.weight", err);
        if (hp.moe()) {
            L.ffn_gate_inp = need(m, p + "ffn_gate_inp.weight", err);
            if (!stream_experts) {
                L.ffn_up_exps = need(m, p + "ffn_up_exps.weight", err);
                L.ffn_gate_exps = need(m, p + "ffn_gate_exps.weight", err);
                L.ffn_down_exps = need(m, p + "ffn_down_exps.weight", err);
            }
        } else {
            L.ffn_gate = need(m, p + "ffn_gate.weight", err);
            L.ffn_up = need(m, p + "ffn_up.weight", err);
            L.ffn_down = need(m, p + "ffn_down.weight", err);
        }
    }
    if (!err.empty()) return false;
    hp.n_vocab = (uint32_t) w.tok_embd->ne[1];
    return true;
}

// Validates expert shapes against the hyperparameters, since the ExpertStore computes byte offsets from them.
bool qwen3_expert_infos(const GgufModel &m, const Qwen3HParams &hp, std::vector<ExpertLayerInfo> &infos, std::string &err) {
    infos.assign(hp.n_layer, ExpertLayerInfo{});
    for (uint32_t l = 0; l < hp.n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ExpertLayerInfo &li = infos[l];
        if (!m.info(p + "ffn_up_exps.weight", li.up) || !m.info(p + "ffn_gate_exps.weight", li.gate) ||
            !m.info(p + "ffn_down_exps.weight", li.down)) {
            err = "missing expert tensors for layer " + std::to_string(l);
            return false;
        }
        if (li.up.ne[2] != hp.n_expert || li.up.ne[0] != hp.n_embd || li.up.ne[1] != hp.n_ff_exp || li.down.ne[0] != hp.n_ff_exp ||
            li.down.ne[1] != hp.n_embd) {
            err = "expert tensor shapes do not match the hyperparameters for layer " + std::to_string(l);
            return false;
        }
    }
    return true;
}

// RMSNorm then elementwise scale by the learned weight.
static ggml_tensor *rms(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Linear projection with optional bias (Qwen2 has Q/K/V biases).
static ggml_tensor *proj(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b, ggml_tensor *x) {
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

// Attention sub-block for layer l, including the residual add. Steps:
//   1. RMSNorm, project to Q (n_head heads) and K, V (n_head_kv heads), optional per-head QK-norm (Qwen3).
//   2. RoPE: rotate Q and K by each token's position so attention scores depend on relative distance.
//   3. Write this step's K and V into their KV cache cells (in.kv_idxs).
//   4. Flash attention of Q over the first n_kv cache cells, with the mask hiding other sequences
//      and future positions; then the output projection.
static ggml_tensor *attn_block(ggml_context *ctx, ggml_cgraph *gf, const Qwen3HParams &hp, const Qwen3Layer &L, const KvCache &kv,
                               uint32_t l, ggml_tensor *x, const Qwen3StepInputs &in, int n_tokens, int n_kv) {
    const int64_t hd = hp.head_dim;
    const int64_t nh = hp.n_head;
    const int64_t nkv = hp.n_head_kv;
    const float kq_scale = 1.0f / sqrtf((float) hd);

    ggml_tensor *h = rms(ctx, x, L.attn_norm, hp.rms_eps);
    ggml_tensor *q = ggml_reshape_3d(ctx, proj(ctx, L.wq, L.bq, h), hd, nh, n_tokens);
    ggml_tensor *k = ggml_reshape_3d(ctx, proj(ctx, L.wk, L.bk, h), hd, nkv, n_tokens);
    ggml_tensor *v = ggml_reshape_3d(ctx, proj(ctx, L.wv, L.bv, h), hd, nkv, n_tokens);
    if (L.q_norm) q = rms(ctx, q, L.q_norm, hp.rms_eps);
    if (L.k_norm) k = rms(ctx, k, L.k_norm, hp.rms_eps);
    // NEOX-style RoPE (rotates the two halves of each head), no context extension: freq_scale 1,
    // ext_factor 0; the YaRN beta values (32, 1) are inert when ext_factor is 0.
    q = ggml_rope_ext(ctx, q, in.pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f,
                      1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, in.pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f,
                      1.0f, 32.0f, 1.0f);

    // set_rows scatters each token's K/V row into its cache column; these nodes must be in the graph
    // explicitly because nothing downstream consumes their result.
    ggml_tensor *k_rows = ggml_view_2d(ctx, k, hd * nkv, n_tokens, k->nb[2], 0);
    ggml_tensor *v_rows = ggml_view_2d(ctx, v, hd * nkv, n_tokens, v->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv.k[l], k_rows, in.kv_idxs));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv.v[l], v_rows, in.kv_idxs));

    // View the cache as [head_dim, n_head_kv, n_kv] and permute to the [head_dim, n_kv, heads] layout
    // flash attention expects. GQA head sharing is handled inside the kernel.
    const size_t row_head = ggml_row_size(kv.type, hd);
    const size_t row_all = ggml_row_size(kv.type, hd * nkv);
    ggml_tensor *K = ggml_view_3d(ctx, kv.k[l], hd, nkv, n_kv, row_head, row_all, 0);
    ggml_tensor *V = ggml_view_3d(ctx, kv.v[l], hd, nkv, n_kv, row_head, row_all, 0);
    ggml_tensor *Q = ggml_permute(ctx, q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    ggml_tensor *attn = ggml_flash_attn_ext(ctx, Q, K, V, in.mask, kq_scale, 0.0f, 0.0f);
    // Accumulate in fp32 even though K/V are fp16, for parity with llama.cpp.
    ggml_prec_set_acc(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(ctx, attn, hd * nh, n_tokens);
    ggml_tensor *o = ggml_mul_mat(ctx, L.wo, attn);
    return ggml_add(ctx, o, x);
}

// MoE router: softmax over all experts' scores, keep the top-k, and renormalize their probabilities to
// sum to 1 (Qwen3-MoE's norm_topk_prob). Outputs expert ids [k, n_tokens] and weights [1, k, n_tokens].
static void router(ggml_context *ctx, ggml_cgraph *gf, const Qwen3HParams &hp, const Qwen3Layer &L, ggml_tensor *hnorm,
                   int n_tokens, ggml_tensor **selected_out, ggml_tensor **weights_out) {
    const int64_t k = hp.n_expert_used;
    ggml_tensor *logits = ggml_mul_mat(ctx, L.ffn_gate_inp, hnorm);
    ggml_tensor *probs = ggml_soft_max(ctx, logits);
    ggml_tensor *selected = ggml_argsort_top_k(ctx, probs, (int) k);
    // The CPU reads the ids back (trace / streaming), so keep them (and their source) from being overwritten.
    ggml_set_output(selected);
    if (selected->view_src) ggml_set_output(selected->view_src);
    ggml_tensor *weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, hp.n_expert, n_tokens), selected);
    weights = ggml_reshape_2d(ctx, weights, k, n_tokens);
    // Clamp the denominator to the smallest normal fp16 (2^-14) to avoid dividing by ~0.
    ggml_tensor *sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(ctx, ggml_div(ctx, weights, sum), 1, k, n_tokens);
    ggml_build_forward_expand(gf, weights);
    *selected_out = selected;
    *weights_out = weights;
}

// Runs the chosen experts: mul_mat_id multiplies each token by the expert matrices named in `ids`
// (expert numbers in resident mode, pool slot numbers in streaming mode), applies SwiGLU, then sums
// the k expert outputs weighted by the router probabilities.
static ggml_tensor *moe_apply(ggml_context *ctx, ggml_tensor *up, ggml_tensor *gate, ggml_tensor *down, ggml_tensor *hnorm,
                              ggml_tensor *ids, ggml_tensor *weights, int rows, int64_t k) {
    const int64_t n_embd = hnorm->ne[0];
    ggml_tensor *x3 = ggml_reshape_3d(ctx, hnorm, n_embd, 1, rows);
    ggml_tensor *u = ggml_mul_mat_id(ctx, up, x3, ids);
    ggml_tensor *g = ggml_mul_mat_id(ctx, gate, x3, ids);
    ggml_tensor *act = ggml_swiglu_split(ctx, g, u);
    ggml_tensor *experts = ggml_mul_mat_id(ctx, down, act, ids);
    experts = ggml_mul(ctx, experts, weights);
    ggml_tensor *out = nullptr;
    for (int64_t i = 0; i < k; i++) {
        ggml_tensor *view = ggml_view_2d(ctx, experts, n_embd, rows, experts->nb[2], i * experts->nb[1]);
        out = out ? ggml_add(ctx, out, view) : view;
    }
    // With k == 1 `out` is still a strided view; make it contiguous for the residual add.
    if (k == 1) out = ggml_cont(ctx, out);
    return out;
}

// Dense SwiGLU feed-forward: down(silu(gate(x)) * up(x)).
static ggml_tensor *dense_ffn(ggml_context *ctx, const Qwen3Layer &L, ggml_tensor *hnorm) {
    ggml_tensor *gate = ggml_mul_mat(ctx, L.ffn_gate, hnorm);
    ggml_tensor *up = ggml_mul_mat(ctx, L.ffn_up, hnorm);
    ggml_tensor *act = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.ffn_down, act);
}

ggml_cgraph *qwen3_build_graph(ggml_context *ctx, size_t graph_size, const Qwen3HParams &hp, const Qwen3Weights &w,
                               const KvCache &kv, const Qwen3StepInputs &in, int n_tokens, int n_kv,
                               std::vector<ggml_tensor *> &selected_out, ggml_tensor **logits_out) {
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, graph_size, false);
    selected_out.clear();
    ggml_tensor *x = ggml_get_rows(ctx, w.tok_embd, in.tokens);
    for (uint32_t l = 0; l < hp.n_layer; l++) {
        const Qwen3Layer &L = w.layers[l];
        x = attn_block(ctx, gf, hp, L, kv, l, x, in, n_tokens, n_kv);
        int rows = n_tokens;
        // After the last attention, only the last token's row matters; skip the FFN for all other rows.
        if (l == hp.n_layer - 1) {
            x = ggml_get_rows(ctx, x, in.out_ids);
            rows = 1;
        }
        ggml_tensor *hn = rms(ctx, x, L.ffn_norm, hp.rms_eps);
        ggml_tensor *ffn = nullptr;
        if (hp.moe()) {
            ggml_tensor *selected = nullptr;
            ggml_tensor *weights = nullptr;
            router(ctx, gf, hp, L, hn, rows, &selected, &weights);
            ffn = moe_apply(ctx, L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps, hn, selected, weights, rows, hp.n_expert_used);
            selected_out.push_back(selected);
        } else {
            ffn = dense_ffn(ctx, L, hn);
        }
        x = ggml_add(ctx, ffn, x);
    }
    x = rms(ctx, x, w.output_norm, hp.rms_eps);
    ggml_tensor *logits = ggml_mul_mat(ctx, w.output, x);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    *logits_out = logits;
    return gf;
}

ggml_cgraph *qwen3_build_segment(ggml_context *ctx, size_t graph_size, const Qwen3HParams &hp, const Qwen3Weights &w,
                                 const KvCache &kv, const Qwen3StepInputs &in, const Qwen3Persist &p, int n_tokens, int n_kv,
                                 int moe_layer, const Qwen3PoolTensors *pool, ggml_tensor **slot_ids_out, int attn_layer,
                                 ggml_tensor **selected_out, ggml_tensor **logits_out) {
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, graph_size, false);
    const int64_t n_embd = hp.n_embd;
    const int64_t k = hp.n_expert_used;
    const bool final = attn_layer < 0;
    const int rows = final ? 1 : n_tokens;
    const size_t row_off = final ? (size_t) (n_tokens - 1) : 0;
    *slot_ids_out = nullptr;
    *selected_out = nullptr;
    *logits_out = nullptr;

    ggml_tensor *x = nullptr;
    // Front half: finish layer moe_layer by running its experts on the state saved by the previous segment.
    // `ids` is a graph input the CPU fills with pool slot numbers after it has loaded the experts.
    if (moe_layer >= 0) {
        ggml_tensor *resid = ggml_view_2d(ctx, p.resid, n_embd, rows, p.resid->nb[1], row_off * p.resid->nb[1]);
        ggml_tensor *hn = ggml_view_2d(ctx, p.hnorm, n_embd, rows, p.hnorm->nb[1], row_off * p.hnorm->nb[1]);
        ggml_tensor *wts = ggml_view_3d(ctx, p.weights, 1, k, rows, p.weights->nb[1], p.weights->nb[2], row_off * p.weights->nb[2]);
        ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, rows);
        ggml_set_name(ids, "slot_ids");
        ggml_set_input(ids);
        *slot_ids_out = ids;
        ggml_tensor *moe = moe_apply(ctx, pool->up, pool->gate, pool->down, hn, ids, wts, rows, k);
        x = ggml_add(ctx, moe, resid);
    } else {
        x = ggml_get_rows(ctx, w.tok_embd, in.tokens);
    }

    // Back half: attention and router of attn_layer. The expert step can't run here because the CPU must
    // first see `selected`, so the state it needs is copied into the persistent tensors for the next segment.
    if (!final) {
        const Qwen3Layer &L = w.layers[attn_layer];
        x = attn_block(ctx, gf, hp, L, kv, (uint32_t) attn_layer, x, in, n_tokens, n_kv);
        ggml_tensor *hn = rms(ctx, x, L.ffn_norm, hp.rms_eps);
        ggml_tensor *selected = nullptr;
        ggml_tensor *weights = nullptr;
        router(ctx, gf, hp, L, hn, n_tokens, &selected, &weights);
        ggml_tensor *resid_dst = ggml_view_2d(ctx, p.resid, n_embd, n_tokens, p.resid->nb[1], 0);
        ggml_tensor *hn_dst = ggml_view_2d(ctx, p.hnorm, n_embd, n_tokens, p.hnorm->nb[1], 0);
        ggml_tensor *w_dst = ggml_view_3d(ctx, p.weights, 1, k, n_tokens, p.weights->nb[1], p.weights->nb[2], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, x, resid_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hn, hn_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, weights, w_dst));
        *selected_out = selected;
    } else {
        x = rms(ctx, x, w.output_norm, hp.rms_eps);
        ggml_tensor *logits = ggml_mul_mat(ctx, w.output, x);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        *logits_out = logits;
    }
    return gf;
}
