// engine.h - the inference engine: everything needed to run one forward pass of the model.
//
// An Engine owns the ggml compute backends (Metal GPU plus CPU), the resident weights, the KV
// cache, the graph scheduler, and, for Mixture-of-Experts models in streaming mode, the
// ExpertStore that pages expert weights in from the SSD. Callers (main.cpp, server.cpp) feed it
// token ids for a sequence and get back the logits of the last token; tokenizing and sampling
// happen outside the engine.
//
// Two execution paths:
//   resident  - all weights in memory; one ggml graph runs the whole model per step.
//   streaming - MoE expert weights stay on disk. The model runs as one graph segment per layer
//               so the CPU can see which experts the router picked, load them into pool slots,
//               and only then run those experts (see qwen3_build_segment and docs/decisions.md).

#pragma once

#include "expert_store.h"
#include "gguf_model.h"
#include "kv_cache.h"
#include "qwen3.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Load-time configuration, filled from the command line in main.cpp.
struct EngineParams {
    std::string model_path;
    // Max tokens the KV cache holds (rounded up to a multiple of 256).
    uint32_t n_ctx = 4096;
    // Max tokens per forward pass; longer prompts are split into chunks of this size.
    uint32_t n_batch = 512;
    // Threads for ops the scheduler places on the CPU backend.
    int n_threads = 4;
    bool verbose = false;
    // Keep MoE experts on disk and stream them (ignored for dense models).
    bool stream_experts = false;
    // Expert pool size in GiB; 0 = size it automatically in Engine::load.
    double pool_gib = 0.0;
    // pread() worker threads used to load experts.
    int io_threads = 4;
    // Metal recommendedMaxWorkingSetSize, used to cap the auto-sized pool.
    uint64_t gpu_cap_bytes = 0;
};

// Counters reported by main.cpp after a run (memory footprint, timings, expert-cache behaviour).
struct EngineStats {
    double load_ms = 0.0;
    double last_decode_ms = 0.0;
    size_t weight_bytes = 0;
    size_t kv_bytes = 0;
    size_t compute_bytes = 0;
    // Total bytes of all expert tensors in the file (what a resident run would hold).
    size_t expert_table_bytes = 0;
    size_t pool_bytes = 0;
    uint32_t pool_slots = 0;
    bool streaming = false;
    ExpertStoreStats experts;
    // Streaming only: graph segments run, and CPU time spent between them (build, alloc, expert I/O).
    uint64_t segments = 0;
    double segment_overhead_ms = 0.0;
};

// Runs the model. Not thread-safe: the server serializes requests with a mutex.
class Engine {
public:
    ~Engine();
    // Opens the GGUF, uploads weights, allocates the KV cache (and expert pools), and pre-reserves compute buffers.
    bool load(const EngineParams &params, std::string &err);
    // Appends tokens to sequence `seq` in the KV cache and runs them through the model in n_batch chunks.
    // `logits` receives n_vocab scores for the last token only.
    bool decode(int32_t seq, const int32_t *tokens, int n_tokens, std::vector<float> &logits, std::string &err);
    // Forget a sequence entirely / keep only its first `keep` tokens (used for KV prefix reuse).
    void remove_seq(int32_t seq);
    void truncate_seq(int32_t seq, int32_t keep) { kv_.truncate_seq(seq, keep); }
    // MoE only: record every layer's chosen expert ids per token to `f` for tools/cache_sim.py.
    void set_routing_trace(FILE *f);

    int32_t seq_len(int32_t seq) const { return kv_.length(seq); }
    uint32_t n_free_tokens() const { return kv_.n_free_cells(); }
    const Qwen3HParams &hparams() const { return hp_; }
    const EngineStats &stats() const { return stats_; }
    std::string describe_pools() const { return stream_ ? store_.describe() : std::string(); }

private:
    // One forward pass over at most n_batch tokens; dispatches to the resident or streaming path.
    bool step(int32_t seq, const int32_t *tokens, int n_tokens, std::vector<float> &logits, std::string &err);
    bool step_resident(int n_tokens, int n_kv, std::vector<float> &logits, std::string &err);
    bool step_streaming(int n_tokens, int n_kv, std::vector<float> &logits, std::string &err);
    // Streaming: run experts of `moe_layer` then attention+router of `attn_layer` as one graph (-1 = none).
    bool run_segment(int moe_layer, int attn_layer, int n_tokens, int n_kv, std::vector<int32_t> &selected,
                     std::vector<float> *logits, std::string &err);
    // Allocates the per-step input tensors (tokens, positions, KV cells, mask) and reserves KV cells for them.
    bool make_step_inputs(int32_t seq, const int32_t *tokens, int n_tokens, int n_kv, std::string &err);
    void free_step_inputs();
    void write_trace_layer(const int32_t *ids, int n_tokens);
    // How many KV cells attention must scan: highest used cell, padded to 256 so graph shapes repeat.
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
    // Splits each graph between GPU and CPU and owns the reusable compute (activation) buffers.
    ggml_backend_sched_t sched_ = nullptr;
    // Max nodes per ggml graph, and the metadata memory needed to build one.
    size_t graph_size_ = 8192;
    size_t graph_mem_size_ = 0;
    // Routing trace output (--trace-routing), or null.
    FILE *trace_ = nullptr;
    // Streaming-mode state: the expert cache and each layer's expert tensor locations in the file.
    bool stream_ = false;
    ExpertStore store_;
    std::vector<ExpertLayerInfo> infos_;
    // Streaming: GPU tensors that carry activations from one graph segment to the next.
    ggml_context *persist_ctx_ = nullptr;
    ggml_backend_buffer_t persist_buf_ = nullptr;
    Qwen3Persist persist_;
    // Per-step input tensors; rebuilt for every step because their shapes change.
    ggml_context *step_ctx_ = nullptr;
    ggml_backend_buffer_t step_buf_ = nullptr;
    Qwen3StepInputs in_;
};
