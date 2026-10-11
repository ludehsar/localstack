#pragma once

#include "expert_store.h"
#include "gguf_model.h"
#include "kv_cache.h"
#include "qwen3.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct EngineParams {
    std::string model_path;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    int n_threads = 4;
    bool verbose = false;
    bool stream_experts = false;
    double pool_gib = 0.0;
    int io_threads = 4;
    uint64_t gpu_cap_bytes = 0;
};

struct EngineStats {
    double load_ms = 0.0;
    double last_decode_ms = 0.0;
    size_t weight_bytes = 0;
    size_t kv_bytes = 0;
    size_t compute_bytes = 0;
    size_t expert_table_bytes = 0;
    size_t pool_bytes = 0;
    uint32_t pool_slots = 0;
    bool streaming = false;
    ExpertStoreStats experts;
    uint64_t segments = 0;
    double segment_overhead_ms = 0.0;
};

class Engine {
public:
    ~Engine();
    bool load(const EngineParams &params, std::string &err);
    bool decode(int32_t seq, const int32_t *tokens, int n_tokens, std::vector<float> &logits, std::string &err);
    void remove_seq(int32_t seq);
    void truncate_seq(int32_t seq, int32_t keep) { kv_.truncate_seq(seq, keep); }
    void set_routing_trace(FILE *f);

    int32_t seq_len(int32_t seq) const { return kv_.length(seq); }
    uint32_t n_free_tokens() const { return kv_.n_free_cells(); }
    const Qwen3HParams &hparams() const { return hp_; }
    const EngineStats &stats() const { return stats_; }
    std::string describe_pools() const { return stream_ ? store_.describe() : std::string(); }

private:
    bool step(int32_t seq, const int32_t *tokens, int n_tokens, std::vector<float> &logits, std::string &err);
    bool step_resident(int n_tokens, int n_kv, std::vector<float> &logits, std::string &err);
    bool step_streaming(int n_tokens, int n_kv, std::vector<float> &logits, std::string &err);
    bool run_segment(int moe_layer, int attn_layer, int n_tokens, int n_kv, std::vector<int32_t> &selected,
                     std::vector<float> *logits, std::string &err);
    bool make_step_inputs(int32_t seq, const int32_t *tokens, int n_tokens, int n_kv, std::string &err);
    void free_step_inputs();
    void write_trace_layer(const int32_t *ids, int n_tokens);
    int n_kv_for(int n_total) const;

    EngineParams params_;
    EngineStats stats_;
    GgufModel model_;
    Qwen3HParams hp_;
    Qwen3Weights w_;
    KvCache kv_;
    ggml_backend_dev_t gpu_dev_ = nullptr;
    ggml_backend_t gpu_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    size_t graph_size_ = 8192;
    size_t graph_mem_size_ = 0;
    FILE *trace_ = nullptr;
    bool stream_ = false;
    ExpertStore store_;
    std::vector<ExpertLayerInfo> infos_;
    ggml_context *persist_ctx_ = nullptr;
    ggml_backend_buffer_t persist_buf_ = nullptr;
    Qwen3Persist persist_;
    ggml_context *step_ctx_ = nullptr;
    ggml_backend_buffer_t step_buf_ = nullptr;
    Qwen3StepInputs in_;
};
