// engine.cpp - Engine implementation: backend setup, memory sizing, and the per-step forward pass.
//
// A decode() call splits its tokens into n_batch chunks; each chunk is one step(). A step reserves KV
// cells, uploads the step inputs, then either runs one whole-model graph (resident) or walks the
// layers as a chain of graph segments, loading routed experts from SSD between segments (streaming).

#include "engine.h"

#include "ggml-cpu.h"
#include "memory_monitor.h"

#include <algorithm>
#include <cstdio>

Engine::~Engine() {
    free_step_inputs();
    if (sched_) ggml_backend_sched_free(sched_);
    if (persist_buf_) ggml_backend_buffer_free(persist_buf_);
    if (persist_ctx_) ggml_free(persist_ctx_);
    kv_.free();
    model_.close();
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
}

// Routing trace format (read by tools/cache_sim.py), all little-endian int32:
//   "LSTKRTE1", n_layer, n_expert, n_expert_used,
//   then per step: n_tokens, then per layer n_tokens * n_expert_used expert ids (-1 = not computed).
static const char TRACE_MAGIC[8] = { 'L', 'S', 'T', 'K', 'R', 'T', 'E', '1' };

bool Engine::load(const EngineParams &params, std::string &err) {
    params_ = params;
    const int64_t t0 = ggml_time_us();

    // Find the Metal GPU device and the CPU device. Weights, KV, and most ops go on the GPU; the
    // scheduler falls back to the CPU for any op the GPU backend does not support.
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                if (!gpu_dev_) gpu_dev_ = dev;
                break;
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                if (!cpu_dev) cpu_dev = dev;
                break;
            default:
                break;
        }
    }
    if (!gpu_dev_ || !cpu_dev) {
        err = "need both a GPU (Metal) and a CPU ggml backend";
        return false;
    }
    gpu_ = ggml_backend_dev_init(gpu_dev_, nullptr);
    cpu_ = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!gpu_ || !cpu_) {
        err = "backend initialization failed";
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu_, params.n_threads);

    if (!model_.open(params.model_path, err)) return false;
    if (!qwen3_load_hparams(model_, hp_, err)) return false;
    // In streaming mode the expert tensors are skipped here and served from the ExpertStore instead.
    stream_ = params.stream_experts && hp_.moe();
    stats_.streaming = stream_;
    if (!model_.load_all(gpu_, err, stream_ ? qwen3_is_expert_tensor : std::function<bool(const std::string &)>())) return false;
    if (!qwen3_map_weights(model_, hp_, w_, stream_, err)) return false;
    stats_.weight_bytes = model_.loaded_bytes();

    // KV cache in fp16, sized to n_ctx rounded up to the 256-cell padding used by n_kv_for().
    const uint32_t kv_size = (params.n_ctx + 255) / 256 * 256;
    if (!kv_.init(gpu_, hp_.n_layer, hp_.n_embd_k_gqa(), hp_.n_embd_v_gqa(), kv_size, GGML_TYPE_F16, err)) return false;
    stats_.kv_bytes = kv_.bytes();

    if (stream_) {
        if (!qwen3_expert_infos(model_, hp_, infos_, err)) return false;
        size_t table = 0;
        for (const ExpertLayerInfo &li : infos_) table += li.up.nbytes + li.gate.nbytes + li.down.nbytes;
        stats_.expert_table_bytes = table;
        // Expert pool size. Unless --pool-gib is given, take the smallest of:
        //   by_table - 45% of all expert bytes (the upper clamp from docs/plan.md section 5.7),
        //   by_cap   - Metal working-set cap minus weights, KV, and 1.5 GiB of compute headroom,
        //   by_free  - free + inactive RAM right now minus a 3 GiB floor left for the OS page cache,
        //              so the pool never pushes macOS into compression or swap.
        size_t pool_bytes = (size_t) (params.pool_gib * 1073741824.0);
        if (pool_bytes == 0) {
            const double by_table = 0.45 * (double) table;
            double by_cap = by_table;
            if (params.gpu_cap_bytes > 0) {
                const double used = (double) stats_.weight_bytes + (double) stats_.kv_bytes + 1.5 * 1073741824.0;
                by_cap = std::max(0.0, (double) params.gpu_cap_bytes - used);
            }
            double by_free = by_table;
            MemorySnapshot mem;
            if (memory_snapshot(mem)) {
                const double floor = 3.0 * 1073741824.0;
                by_free = std::max(0.0, (double) mem.available() - floor);
                if (params.verbose) fprintf(stderr, "memory now: %s\n", memory_describe(mem).c_str());
            }
            pool_bytes = (size_t) std::min(by_table, std::min(by_cap, by_free));
            if (params.verbose) {
                fprintf(stderr, "pool sizing: table %.2f GiB -> by_table %.2f, by_gpu_cap %.2f, by_free_memory %.2f -> %.2f GiB\n",
                        table / 1073741824.0, by_table / 1073741824.0, by_cap / 1073741824.0, by_free / 1073741824.0,
                        pool_bytes / 1073741824.0);
            }
        }
        // Clamp. ExpertStore::init still raises each pool to its minimum slot count, so a tiny value is safe.
        pool_bytes = std::max<size_t>(pool_bytes, 16u << 10);
        pool_bytes = std::min<size_t>(pool_bytes, (size_t) 9 << 30);
        // Minimum slots per pool: a full layer's experts, because a prefill batch can route to all of them at once.
        if (!store_.init(gpu_dev_, gpu_, model_.fd, hp_.n_layer, hp_.n_expert, hp_.n_embd, hp_.n_ff_exp, infos_, pool_bytes,
                         std::max(hp_.n_expert, 2 * hp_.n_expert_used + 2), params.io_threads, err)) {
            return false;
        }
        stats_.pool_bytes = store_.total_bytes();
        stats_.pool_slots = store_.total_slots();
        if (params.verbose) fprintf(stderr, "expert pools:\n%s", store_.describe().c_str());

        // Activations that cross a segment boundary, sized for a full n_batch chunk. They live in their own
        // GPU buffer because the scheduler's compute buffers are reused (overwritten) by every graph.
        ggml_init_params pp = { ggml_tensor_overhead() * 4, nullptr, true };
        persist_ctx_ = ggml_init(pp);
        persist_.resid = ggml_new_tensor_2d(persist_ctx_, GGML_TYPE_F32, hp_.n_embd, params.n_batch);
        persist_.hnorm = ggml_new_tensor_2d(persist_ctx_, GGML_TYPE_F32, hp_.n_embd, params.n_batch);
        persist_.weights = ggml_new_tensor_3d(persist_ctx_, GGML_TYPE_F32, 1, hp_.n_expert_used, params.n_batch);
        ggml_set_name(persist_.resid, "persist_resid");
        ggml_set_name(persist_.hnorm, "persist_hnorm");
        ggml_set_name(persist_.weights, "persist_weights");
        persist_buf_ = ggml_backend_alloc_ctx_tensors(persist_ctx_, gpu_);
        if (!persist_buf_) {
            err = "failed to allocate persistent activations";
            return false;
        }
    }

    // Build the largest graph we will ever run (n_batch tokens attending over the whole cache) once, so
    // the scheduler sizes its compute buffers up front and later steps never reallocate.
    graph_mem_size_ = ggml_tensor_overhead() * graph_size_ * 2 + ggml_graph_overhead_custom(graph_size_, false);
    ggml_backend_t backends[2] = { gpu_, cpu_ };
    sched_ = ggml_backend_sched_new(backends, nullptr, 2, graph_size_, false, true);

    std::vector<int32_t> dummy(params.n_batch, 0);
    if (!make_step_inputs(-1, dummy.data(), (int) params.n_batch, (int) kv_size, err)) return false;
    ggml_init_params gp = { graph_mem_size_, nullptr, true };
    ggml_context *ctx = ggml_init(gp);
    ggml_cgraph *measure = nullptr;
    std::vector<ggml_tensor *> sel;
    ggml_tensor *logits = nullptr;
    if (stream_) {
        const ExpertPool &P = store_.pool_for(0);
        Qwen3PoolTensors pt = { P.up, P.gate, P.down };
        ggml_tensor *ids = nullptr;
        ggml_tensor *selected = nullptr;
        measure = qwen3_build_segment(ctx, graph_size_, hp_, w_, kv_, in_, persist_, (int) params.n_batch, (int) kv_size, 0, &pt,
                                      &ids, std::min<int>(1, (int) hp_.n_layer - 1), &selected, &logits);
    } else {
        measure = qwen3_build_graph(ctx, graph_size_, hp_, w_, kv_, in_, (int) params.n_batch, (int) kv_size, sel, &logits);
    }
    const bool reserved = ggml_backend_sched_reserve(sched_, measure);
    ggml_free(ctx);
    free_step_inputs();
    if (!reserved) {
        err = "failed to reserve compute buffers";
        return false;
    }
    stats_.compute_bytes = ggml_backend_sched_get_buffer_size(sched_, gpu_) + ggml_backend_sched_get_buffer_size(sched_, cpu_);
    stats_.load_ms = (ggml_time_us() - t0) / 1000.0;
    return true;
}

void Engine::remove_seq(int32_t seq) {
    kv_.remove_seq(seq);
}

void Engine::set_routing_trace(FILE *f) {
    trace_ = f;
    if (!trace_ || !hp_.moe()) return;
    const int32_t header[3] = { (int32_t) hp_.n_layer, (int32_t) hp_.n_expert, (int32_t) hp_.n_expert_used };
    fwrite(TRACE_MAGIC, 1, 8, trace_);
    fwrite(header, sizeof(int32_t), 3, trace_);
}

void Engine::write_trace_layer(const int32_t *ids, int n_tokens) {
    fwrite(ids, sizeof(int32_t), (size_t) hp_.n_expert_used * n_tokens, trace_);
}

// Attention only needs to scan cells [0, used_hwm). Padding to 256 (at least 256, at most the cache size)
// keeps the number of distinct graph shapes small; the mask hides the padding cells.
int Engine::n_kv_for(int n_total) const {
    const int padded = (n_total + 255) / 256 * 256;
    return std::min<int>((int) kv_.size, std::max(256, padded));
}

// Reserves KV cells for `tokens` in `seq` and uploads the step's input tensors to the GPU.
// seq < 0 is the load-time dry run: shapes only, no KV allocation and no data.
bool Engine::make_step_inputs(int32_t seq, const int32_t *tokens, int n_tokens, int n_kv, std::string &err) {
    free_step_inputs();
    std::vector<int64_t> cells(n_tokens);
    std::vector<int32_t> pos(n_tokens);
    if (seq >= 0) {
        if (!kv_.alloc(seq, n_tokens, cells, pos)) {
            err = "KV cache is full";
            return false;
        }
    } else {
        for (int i = 0; i < n_tokens; i++) {
            cells[i] = i;
            pos[i] = i;
        }
    }
    ggml_init_params sp = { ggml_tensor_overhead() * 8, nullptr, true };
    step_ctx_ = ggml_init(sp);
    in_.tokens = ggml_new_tensor_1d(step_ctx_, GGML_TYPE_I32, n_tokens);
    in_.pos = ggml_new_tensor_1d(step_ctx_, GGML_TYPE_I32, n_tokens);
    in_.kv_idxs = ggml_new_tensor_1d(step_ctx_, GGML_TYPE_I64, n_tokens);
    in_.mask = ggml_new_tensor_2d(step_ctx_, GGML_TYPE_F16, n_kv, n_tokens);
    in_.out_ids = ggml_new_tensor_1d(step_ctx_, GGML_TYPE_I32, 1);
    ggml_set_name(in_.tokens, "inp_tokens");
    ggml_set_name(in_.pos, "inp_pos");
    ggml_set_name(in_.kv_idxs, "inp_kv_idxs");
    ggml_set_name(in_.mask, "inp_mask");
    ggml_set_name(in_.out_ids, "inp_out_ids");
    step_buf_ = ggml_backend_alloc_ctx_tensors(step_ctx_, gpu_);
    if (!step_buf_) {
        err = "failed to allocate step inputs";
        return false;
    }
    if (seq < 0) return true;
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n_tokens);
    kv_.fill_mask(seq, pos.data(), n_tokens, n_kv, mask.data());
    // Only the last token's logits are needed to pick the next token.
    const int32_t out_id = n_tokens - 1;
    ggml_backend_tensor_set(in_.tokens, tokens, 0, sizeof(int32_t) * n_tokens);
    ggml_backend_tensor_set(in_.pos, pos.data(), 0, sizeof(int32_t) * n_tokens);
    ggml_backend_tensor_set(in_.kv_idxs, cells.data(), 0, sizeof(int64_t) * n_tokens);
    ggml_backend_tensor_set(in_.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(in_.out_ids, &out_id, 0, sizeof(int32_t));
    return true;
}

void Engine::free_step_inputs() {
    if (step_buf_) ggml_backend_buffer_free(step_buf_);
    if (step_ctx_) ggml_free(step_ctx_);
    step_buf_ = nullptr;
    step_ctx_ = nullptr;
    in_ = Qwen3StepInputs{};
}

// Public entry point. Long prompts are split into n_batch chunks; only the last chunk's logits survive.
bool Engine::decode(int32_t seq, const int32_t *tokens, int n_tokens, std::vector<float> &logits, std::string &err) {
    const int64_t t0 = ggml_time_us();
    for (int done = 0; done < n_tokens;) {
        const int n = std::min<int>((int) params_.n_batch, n_tokens - done);
        if (!step(seq, tokens + done, n, logits, err)) return false;
        done += n;
    }
    stats_.last_decode_ms = (ggml_time_us() - t0) / 1000.0;
    if (stream_) stats_.experts = store_.stats();
    return true;
}

bool Engine::step(int32_t seq, const int32_t *tokens, int n_tokens, std::vector<float> &logits, std::string &err) {
    const int n_kv = n_kv_for((int) kv_.used_hwm + n_tokens);
    if (!make_step_inputs(seq, tokens, n_tokens, n_kv, err)) return false;
    const bool ok = stream_ ? step_streaming(n_tokens, n_kv, logits, err) : step_resident(n_tokens, n_kv, logits, err);
    free_step_inputs();
    return ok;
}

// Copies the router's chosen expert ids ([k, rows] int32 on the GPU, possibly strided) into `dst`.
static void read_selected(ggml_tensor *sel, int rows, int k, int32_t *dst) {
    for (int t = 0; t < rows; t++) {
        ggml_backend_tensor_get(sel, dst + (size_t) t * k, (size_t) t * sel->nb[1], sizeof(int32_t) * k);
    }
}

// Resident path: build the full-model graph, let the scheduler place and run it, read back the logits.
bool Engine::step_resident(int n_tokens, int n_kv, std::vector<float> &logits, std::string &err) {
    ggml_init_params gp = { graph_mem_size_, nullptr, true };
    ggml_context *ctx = ggml_init(gp);
    std::vector<ggml_tensor *> selected;
    ggml_tensor *logits_t = nullptr;
    ggml_cgraph *gf = qwen3_build_graph(ctx, graph_size_, hp_, w_, kv_, in_, n_tokens, n_kv, selected, &logits_t);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "failed to allocate the compute graph";
        return false;
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "graph compute failed";
        return false;
    }
    logits.resize(hp_.n_vocab);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, sizeof(float) * hp_.n_vocab);
    if (trace_ && hp_.moe()) {
        const int32_t n = n_tokens;
        fwrite(&n, sizeof(int32_t), 1, trace_);
        std::vector<int32_t> ids((size_t) hp_.n_expert_used * n_tokens);
        // The last layer only runs its FFN for the output row, so its trace has one real row and -1 elsewhere.
        for (ggml_tensor *sel : selected) {
            const int rows = (int) sel->ne[1];
            std::fill(ids.begin(), ids.end(), -1);
            if (rows == n_tokens) {
                read_selected(sel, rows, (int) hp_.n_expert_used, ids.data());
            } else {
                read_selected(sel, 1, (int) hp_.n_expert_used, ids.data() + (size_t) (n_tokens - 1) * hp_.n_expert_used);
            }
            write_trace_layer(ids.data(), n_tokens);
        }
        fflush(trace_);
    }
    ggml_free(ctx);
    return true;
}

// Streaming path. For L layers the step runs L + 1 segments:
//   [embed + attn/router 0], [experts 0 + attn/router 1], ..., [experts L-1 + output head]
// After each segment the CPU reads which experts were selected, so run_segment can load them before
// the next segment, which is the one that uses them.
bool Engine::step_streaming(int n_tokens, int n_kv, std::vector<float> &logits, std::string &err) {
    if (trace_) {
        const int32_t n = n_tokens;
        fwrite(&n, sizeof(int32_t), 1, trace_);
    }
    std::vector<int32_t> selected((size_t) hp_.n_expert_used * n_tokens, 0);
    if (!run_segment(-1, 0, n_tokens, n_kv, selected, nullptr, err)) return false;
    for (uint32_t l = 1; l < hp_.n_layer; l++) {
        if (!run_segment((int) l - 1, (int) l, n_tokens, n_kv, selected, nullptr, err)) return false;
    }
    if (!run_segment((int) hp_.n_layer - 1, -1, n_tokens, n_kv, selected, &logits, err)) return false;
    if (trace_) fflush(trace_);
    return true;
}

// Builds and runs one segment. Between graph allocation and compute, the CPU translates the previous
// router's expert ids into pool slots (loading misses from the SSD) and uploads them as `slot_ids`.
bool Engine::run_segment(int moe_layer, int attn_layer, int n_tokens, int n_kv, std::vector<int32_t> &selected,
                         std::vector<float> *logits, std::string &err) {
    const int64_t t0 = ggml_time_us();
    const int k = (int) hp_.n_expert_used;
    ggml_init_params gp = { graph_mem_size_, nullptr, true };
    ggml_context *ctx = ggml_init(gp);
    Qwen3PoolTensors pt;
    if (moe_layer >= 0) {
        const ExpertPool &P = store_.pool_for(moe_layer);
        pt = { P.up, P.gate, P.down };
    }
    ggml_tensor *slot_ids = nullptr;
    ggml_tensor *sel_t = nullptr;
    ggml_tensor *logits_t = nullptr;
    ggml_cgraph *gf = qwen3_build_segment(ctx, graph_size_, hp_, w_, kv_, in_, persist_, n_tokens, n_kv, moe_layer,
                                          moe_layer >= 0 ? &pt : nullptr, &slot_ids, attn_layer, &sel_t, &logits_t);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "failed to allocate a graph segment";
        return false;
    }
    if (slot_ids) {
        // The final segment only computes the last token, so only its row of expert ids is needed.
        const int rows = attn_layer < 0 ? 1 : n_tokens;
        const int32_t *experts = selected.data() + (attn_layer < 0 ? (size_t) (n_tokens - 1) * k : 0);
        std::vector<int32_t> slots((size_t) k * rows);
        if (!store_.acquire(moe_layer, experts, k * rows, slots.data(), err)) {
            store_.release_all();
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(slot_ids, slots.data(), 0, slots.size() * sizeof(int32_t));
    }
    // Everything up to here (graph build, allocation, expert I/O) is per-segment overhead.
    const int64_t t1 = ggml_time_us();
    const ggml_status st = ggml_backend_sched_graph_compute(sched_, gf);
    // Compute is synchronous, so the experts are no longer needed and can become eviction candidates.
    store_.release_all();
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "graph segment compute failed";
        return false;
    }
    if (sel_t) {
        read_selected(sel_t, n_tokens, k, selected.data());
        if (trace_) write_trace_layer(selected.data(), n_tokens);
    }
    if (logits_t && logits) {
        logits->resize(hp_.n_vocab);
        ggml_backend_tensor_get(logits_t, logits->data(), 0, sizeof(float) * hp_.n_vocab);
    }
    ggml_free(ctx);
    stats_.segments++;
    stats_.segment_overhead_ms += (t1 - t0) / 1000.0;
    return true;
}
