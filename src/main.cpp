// main.cpp - the lstk command-line entry point.
//
// Parses flags, loads the tokenizer and the Engine, then runs one mode:
//   (default)      complete --prompt, optionally comparing every step against llama.cpp (--parity)
//   --chat         interactive multi-turn chat in the terminal
//   --serve        OpenAI-compatible HTTP server (server.cpp)
//   --bench P G    prompt-processing and generation speed
//   --dump-ref     run llama.cpp alone and record its logits (no Engine is loaded)
//   --compare-ref  replay a --dump-ref file through the Engine and report the logit differences
// Afterwards it prints memory-pressure and expert-cache statistics.

#include "engine.h"
#include "memory_monitor.h"
#include "sampler.h"
#include "server.h"
#include "tokenizer.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Defined in metal_info.mm.
extern "C" uint64_t lstk_metal_recommended_working_set(void);

// Parsed command-line options; see usage() for meanings and defaults.
struct Args {
    std::string model;
    std::string prompt = "The capital of France is";
    std::string system;
    std::string dump_ref;
    std::string compare_ref;
    std::string trace_routing;
    std::vector<int32_t> tokens;
    int print_logits = 0;
    int n_predict = 64;
    int bench_prompt = 0;
    int bench_gen = 0;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    bool chat = false;
    bool serve = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    bool parity = false;
    bool verbose = false;
    bool stream_experts = false;
    double pool_gib = 0.0;
    int io_threads = 4;
    SamplerParams sampling;
};

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s --model FILE.gguf [options]\n"
            "  --prompt TEXT        single-shot completion prompt (default mode)\n"
            "  --chat               interactive chat using the model's chat template\n"
            "  --serve              OpenAI-compatible HTTP server (--host, --port; default 127.0.0.1:8080)\n"
            "  --system TEXT        system prompt for chat mode\n"
            "  --n-predict N        max tokens to generate (default 64, chat 512)\n"
            "  --n-ctx N            context size (default 4096)   --n-batch N  prefill batch (default 512)\n"
            "  --temp F --top-k N --top-p F --min-p F --repeat-penalty F --seed N   sampling (temp 0 = greedy)\n"
            "  --parity             run llama.cpp in-process and compare logits every step (greedy)\n"
            "  --dump-ref FILE      run only llama.cpp and record per-step logits and greedy tokens\n"
            "  --compare-ref FILE   run only this engine on the recorded tokens and compare logits\n"
            "  --trace-routing FILE record per-layer expert ids per token (MoE models) for tools/cache_sim.py\n"
            "  --bench P G          warm up, then time a P-token prompt and G greedy decode tokens\n"
            "  --stream-experts     keep MoE experts on disk and stream them through a slot pool (MoE models)\n"
            "  --pool-gib F         expert pool size in GiB (default: auto from the Metal working-set cap)\n"
            "  --io-threads N       pread threads for expert loads (default 4)\n"
            "  --tokens a,b,c       raw prompt token ids (no tokenizer needed; outputs are printed as ids)\n"
            "  --print-logits N     print the top-N logits after the prompt and after every step\n"
            "  --verbose\n",
            prog);
}

static bool parse_args(int argc, char **argv, Args &a) {
    bool n_predict_set = false;
    for (int i = 1; i < argc; i++) {
        const std::string k = argv[i];
        auto next = [&](std::string &dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (k == "--model" || k == "-m") { if (!next(v)) return false; a.model = v; }
        else if (k == "--prompt" || k == "-p") { if (!next(v)) return false; a.prompt = v; }
        else if (k == "--system") { if (!next(v)) return false; a.system = v; }
        else if (k == "--n-predict" || k == "-n") { if (!next(v)) return false; a.n_predict = atoi(v.c_str()); n_predict_set = true; }
        else if (k == "--n-ctx" || k == "-c") { if (!next(v)) return false; a.n_ctx = (uint32_t) atoi(v.c_str()); }
        else if (k == "--n-batch" || k == "-b") { if (!next(v)) return false; a.n_batch = (uint32_t) atoi(v.c_str()); }
        else if (k == "--temp") { if (!next(v)) return false; a.sampling.temperature = (float) atof(v.c_str()); }
        else if (k == "--top-k") { if (!next(v)) return false; a.sampling.top_k = atoi(v.c_str()); }
        else if (k == "--top-p") { if (!next(v)) return false; a.sampling.top_p = (float) atof(v.c_str()); }
        else if (k == "--min-p") { if (!next(v)) return false; a.sampling.min_p = (float) atof(v.c_str()); }
        else if (k == "--repeat-penalty") { if (!next(v)) return false; a.sampling.repeat_penalty = (float) atof(v.c_str()); }
        else if (k == "--seed") { if (!next(v)) return false; a.sampling.seed = (uint32_t) strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--dump-ref") { if (!next(v)) return false; a.dump_ref = v; }
        else if (k == "--compare-ref") { if (!next(v)) return false; a.compare_ref = v; }
        else if (k == "--trace-routing") { if (!next(v)) return false; a.trace_routing = v; }
        else if (k == "--bench") {
            std::string g;
            if (!next(v) || !next(g)) return false;
            a.bench_prompt = atoi(v.c_str());
            a.bench_gen = atoi(g.c_str());
        }
        else if (k == "--stream-experts") a.stream_experts = true;
        else if (k == "--pool-gib") { if (!next(v)) return false; a.pool_gib = atof(v.c_str()); }
        else if (k == "--io-threads") { if (!next(v)) return false; a.io_threads = atoi(v.c_str()); }
        else if (k == "--print-logits") { if (!next(v)) return false; a.print_logits = atoi(v.c_str()); }
        else if (k == "--tokens") {
            if (!next(v)) return false;
            size_t start = 0;
            while (start <= v.size()) {
                size_t comma = v.find(',', start);
                if (comma == std::string::npos) comma = v.size();
                if (comma > start) a.tokens.push_back(atoi(v.substr(start, comma - start).c_str()));
                start = comma + 1;
            }
        }
        else if (k == "--chat") a.chat = true;
        else if (k == "--serve") a.serve = true;
        else if (k == "--host") { if (!next(v)) return false; a.host = v; }
        else if (k == "--port") { if (!next(v)) return false; a.port = atoi(v.c_str()); }
        else if (k == "--parity") a.parity = true;
        else if (k == "--verbose" || k == "-v") a.verbose = true;
        else return false;
    }
    if ((a.chat || a.serve) && !n_predict_set) a.n_predict = 512;
    return !a.model.empty();
}

static bool g_verbose = false;

// ggml/llama.cpp log hook: only warnings and errors unless --verbose.
static void quiet_log(ggml_log_level level, const char *text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static int argmax(const float *v, int n) {
    return (int) (std::max_element(v, v + n) - v);
}

// Prints the n highest logits as id:value (used by --print-logits and the streaming tests).
static void print_top(const std::vector<float> &logits, int n) {
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(n, idx.size()), idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    printf("top%d:", n);
    for (int i = 0; i < n && i < (int) idx.size(); i++) printf(" %d:%.5f", idx[i], logits[idx[i]]);
    printf("\n");
}

// Writes token pieces to stdout as they arrive, holding back a trailing incomplete UTF-8 character
// until the next piece completes it.
struct Utf8Printer {
    std::string pending;
    void write(const std::string &piece) {
        pending += piece;
        size_t cut = pending.size();
        for (size_t back = 1; back <= 4 && back <= pending.size(); back++) {
            const unsigned char c = (unsigned char) pending[pending.size() - back];
            if ((c & 0xC0) == 0x80) continue;
            const size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
            if (back < need) cut = pending.size() - back;
            break;
        }
        fwrite(pending.data(), 1, cut, stdout);
        fflush(stdout);
        pending.erase(0, cut);
    }
    void flush() {
        fwrite(pending.data(), 1, pending.size(), stdout);
        fflush(stdout);
        pending.clear();
    }
};

// How far our logits are from the reference: max and RMS absolute difference, and both argmaxes.
struct Diff {
    double max_abs = 0.0;
    double rms = 0.0;
    int ref_arg = 0;
    int our_arg = 0;
};

static Diff diff_logits(const float *ref, const float *ours, int n) {
    Diff d;
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        const double x = fabs((double) ref[i] - (double) ours[i]);
        d.max_abs = std::max(d.max_abs, x);
        sum_sq += x * x;
    }
    d.rms = sqrt(sum_sq / n);
    d.ref_arg = argmax(ref, n);
    d.our_arg = argmax(ours, n);
    return d;
}

static void report(const char *label, const Diff &d) {
    printf("%s max_abs=%.5f rms=%.6f argmax ref=%d ours=%d %s\n", label, d.max_abs, d.rms, d.ref_arg, d.our_arg,
           d.ref_arg == d.our_arg ? "MATCH" : "MISMATCH");
}

// llama.cpp's own implementation of the same model, run in-process as ground truth for correctness checks.
struct RefModel {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    int n_vocab = 0;

    bool load(const Args &a) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) return false;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = a.n_ctx;
        cp.n_batch = a.n_batch;
        cp.n_ubatch = a.n_batch;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.no_perf = true;
        ctx = llama_init_from_model(model, cp);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        return ctx != nullptr;
    }
    bool decode(int32_t *tokens, int n) { return llama_decode(ctx, llama_batch_get_one(tokens, n)) == 0; }
    const float *logits() { return llama_get_logits_ith(ctx, -1); }
    ~RefModel() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

// Reference file format: "LSTKREF1", int32 n_vocab, n_steps, n_prompt, n_prompt prompt token ids,
// then per step: int32 greedy next token, float[n_vocab] logits.
static const char REF_MAGIC[8] = { 'L', 'S', 'T', 'K', 'R', 'E', 'F', '1' };

// --dump-ref: greedy-decode with llama.cpp and record every step's logits.
static int run_dump_ref(const Args &a, Tokenizer &tok) {
    RefModel ref;
    if (!ref.load(a)) {
        fprintf(stderr, "error: failed to load the llama.cpp reference model\n");
        return 1;
    }
    std::vector<int32_t> prompt = tok.encode(a.prompt, true, true);
    FILE *f = fopen(a.dump_ref.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "error: cannot write %s\n", a.dump_ref.c_str());
        return 1;
    }
    const int32_t n_steps = a.n_predict + 1;
    fwrite(REF_MAGIC, 1, 8, f);
    fwrite(&ref.n_vocab, sizeof(int32_t), 1, f);
    fwrite(&n_steps, sizeof(int32_t), 1, f);
    const int32_t n_prompt = (int32_t) prompt.size();
    fwrite(&n_prompt, sizeof(int32_t), 1, f);
    fwrite(prompt.data(), sizeof(int32_t), prompt.size(), f);
    if (!ref.decode(prompt.data(), n_prompt)) return 1;
    printf("%s", a.prompt.c_str());
    for (int s = 0; s < n_steps; s++) {
        const float *l = ref.logits();
        int32_t next = argmax(l, ref.n_vocab);
        fwrite(&next, sizeof(int32_t), 1, f);
        fwrite(l, sizeof(float), ref.n_vocab, f);
        if (s + 1 < n_steps) {
            printf("%s", tok.piece(next).c_str());
            if (!ref.decode(&next, 1)) return 1;
        }
    }
    printf("\n");
    fclose(f);
    fprintf(stderr, "wrote %d steps of reference logits to %s\n", n_steps, a.dump_ref.c_str());
    return 0;
}

// --compare-ref: feed the recorded prompt and the recorded greedy tokens through our Engine, so both
// implementations see identical inputs, and compare logits at every step.
static int run_compare_ref(const Args &a, Tokenizer &tok, Engine &engine) {
    FILE *f = fopen(a.compare_ref.c_str(), "rb");
    char magic[8];
    int32_t n_vocab = 0, n_steps = 0, n_prompt = 0;
    if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, REF_MAGIC, 8) != 0 || fread(&n_vocab, 4, 1, f) != 1 ||
        fread(&n_steps, 4, 1, f) != 1 || fread(&n_prompt, 4, 1, f) != 1) {
        fprintf(stderr, "error: bad reference file %s\n", a.compare_ref.c_str());
        return 1;
    }
    if (n_vocab != (int32_t) engine.hparams().n_vocab) {
        fprintf(stderr, "error: vocab size mismatch %d vs %u\n", n_vocab, engine.hparams().n_vocab);
        return 1;
    }
    std::vector<int32_t> prompt(n_prompt);
    if (fread(prompt.data(), sizeof(int32_t), n_prompt, f) != (size_t) n_prompt) return 1;
    std::string err;
    std::vector<float> ours;
    if (!engine.decode(0, prompt.data(), n_prompt, ours, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> ref(n_vocab);
    Diff worst;
    int mismatches = 0;
    for (int s = 0; s < n_steps; s++) {
        int32_t next = 0;
        if (fread(&next, 4, 1, f) != 1 || fread(ref.data(), sizeof(float), n_vocab, f) != (size_t) n_vocab) break;
        Diff d = diff_logits(ref.data(), ours.data(), n_vocab);
        if (d.ref_arg != d.our_arg) mismatches++;
        if (d.max_abs > worst.max_abs) worst = d;
        if (a.verbose || d.ref_arg != d.our_arg) {
            char label[32];
            snprintf(label, sizeof(label), "step %-4d", s);
            report(label, d);
        }
        if (s + 1 < n_steps) {
            printf("%s", tok.piece(next).c_str());
            fflush(stdout);
            if (!engine.decode(0, &next, 1, ours, err)) {
                fprintf(stderr, "\nerror: %s\n", err.c_str());
                return 1;
            }
        }
    }
    printf("\n");
    fclose(f);
    printf("parity vs llama.cpp over %d steps: worst max_abs=%.5f (rms %.6f), argmax mismatches=%d\n", n_steps, worst.max_abs,
           worst.rms, mismatches);
    return 0;
}

// --bench: warm up (first Metal pipeline compiles are slow), then time P prompt tokens and G greedy
// tokens. The prompt is synthetic but deterministic token ids spread across the vocabulary.
static int run_bench(const Args &a, Engine &engine) {
    std::string err;
    std::vector<float> logits;
    std::vector<int32_t> prompt(a.bench_prompt);
    for (int i = 0; i < a.bench_prompt; i++) prompt[i] = 1000 + (i * 7919) % 20000;
    const int warm = std::min(a.bench_prompt, 32);
    if (!engine.decode(0, prompt.data(), warm, logits, err) || !engine.decode(0, prompt.data(), 1, logits, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    engine.remove_seq(0);
    const int64_t t0 = ggml_time_us();
    if (!engine.decode(0, prompt.data(), a.bench_prompt, logits, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const double pp_ms = (ggml_time_us() - t0) / 1000.0;
    const int64_t t1 = ggml_time_us();
    for (int i = 0; i < a.bench_gen; i++) {
        int32_t next = argmax(logits.data(), (int) engine.hparams().n_vocab);
        if (!engine.decode(0, &next, 1, logits, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
    }
    const double tg_ms = (ggml_time_us() - t1) / 1000.0;
    printf("bench: pp%d %.1f tok/s (%.0f ms)   tg%d %.2f tok/s (%.1f ms/token)\n", a.bench_prompt, a.bench_prompt * 1000.0 / pp_ms,
           pp_ms, a.bench_gen, a.bench_gen * 1000.0 / tg_ms, tg_ms / std::max(1, a.bench_gen));
    return 0;
}

// --chat: each turn re-applies the chat template to the whole history and feeds only the tokens
// beyond what the KV cache already holds. If the re-tokenized history diverges from what was fed,
// the cache is cleared and the whole history recomputed.
static int run_chat(const Args &a, Tokenizer &tok, Engine &engine) {
    std::vector<std::pair<std::string, std::string>> msgs;
    if (!a.system.empty()) msgs.push_back({ "system", a.system });
    std::vector<int32_t> fed;
    Sampler sampler(a.sampling);
    std::string err;
    std::vector<float> logits;
    fprintf(stderr, "chat ready: type a message, /reset to clear, /exit to quit\n");
    std::string line;
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            engine.remove_seq(0);
            fed.clear();
            sampler.reset();
            msgs.clear();
            if (!a.system.empty()) msgs.push_back({ "system", a.system });
            continue;
        }
        if (line.empty()) continue;
        msgs.push_back({ "user", line });
        const std::string formatted = tok.apply_template(msgs, true);
        std::vector<int32_t> tokens = tok.encode(formatted, true, true);
        size_t common = 0;
        while (common < fed.size() && common < tokens.size() && fed[common] == tokens[common]) common++;
        if (common < fed.size()) {
            engine.remove_seq(0);
            fed.clear();
            common = 0;
            if (a.verbose) fprintf(stderr, "[history re-tokenized differently, recomputing %zu tokens]\n", tokens.size());
        }
        const std::vector<int32_t> suffix(tokens.begin() + (long) common, tokens.end());
        if (suffix.empty()) {
            fprintf(stderr, "[nothing new to process]\n");
            continue;
        }
        const int64_t t0 = ggml_time_us();
        if (!engine.decode(0, suffix.data(), (int) suffix.size(), logits, err)) {
            fprintf(stderr, "error: %s (use /reset)\n", err.c_str());
            msgs.pop_back();
            continue;
        }
        const double prompt_ms = (ggml_time_us() - t0) / 1000.0;
        fed.insert(fed.end(), suffix.begin(), suffix.end());
        for (int32_t t : suffix) sampler.accept(t);

        Utf8Printer out;
        std::string reply;
        int generated = 0;
        const int64_t t1 = ggml_time_us();
        for (int i = 0; i < a.n_predict; i++) {
            const int32_t next = sampler.sample(logits);
            sampler.accept(next);
            fed.push_back(next);
            // Feed the EOG token so the KV matches the template's end-of-turn marker next turn.
            if (tok.is_eog(next)) {
                engine.decode(0, &next, 1, logits, err);
                break;
            }
            const std::string piece = tok.piece(next);
            reply += piece;
            out.write(piece);
            generated++;
            if (!engine.decode(0, &next, 1, logits, err)) {
                fprintf(stderr, "\nerror: %s (use /reset)\n", err.c_str());
                break;
            }
        }
        out.flush();
        const double gen_ms = (ggml_time_us() - t1) / 1000.0;
        msgs.push_back({ "assistant", reply });
        fprintf(stderr, "\n[%zu prompt tokens in %.0f ms, %d generated at %.1f tok/s, context %d/%u]\n", suffix.size(), prompt_ms,
                generated, generated * 1000.0 / std::max(gen_ms, 1.0), engine.seq_len(0), a.n_ctx);
    }
    return 0;
}

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    g_verbose = args.verbose;
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    std::string err;
    Tokenizer tok;
    const bool raw = !args.tokens.empty();
    if (!raw && !tok.load(args.model, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (!args.dump_ref.empty()) {
        const int rc = run_dump_ref(args, tok);
        llama_backend_free();
        return rc;
    }

    // Everything below needs our Engine.
    EngineParams ep;
    ep.model_path = args.model;
    ep.n_ctx = args.n_ctx;
    ep.n_batch = args.n_batch;
    ep.verbose = args.verbose;
    ep.stream_experts = args.stream_experts;
    ep.pool_gib = args.pool_gib;
    ep.io_threads = args.io_threads;
    ep.gpu_cap_bytes = lstk_metal_recommended_working_set();
    Engine engine;
    if (!engine.load(ep, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const Qwen3HParams &hp = engine.hparams();
    const EngineStats &st = engine.stats();
    fprintf(stderr, "model: %s  layers=%u embd=%u heads=%u/%u head_dim=%u ff=%u vocab=%u", args.model.c_str(), hp.n_layer,
            hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_ff, hp.n_vocab);
    if (hp.moe()) fprintf(stderr, "  experts=%u used=%u ff_exp=%u", hp.n_expert, hp.n_expert_used, hp.n_ff_exp);
    fprintf(stderr, "\n");
    MemorySnapshot mem_start;
    const bool have_mem = memory_snapshot(mem_start);
    const int64_t t_run0 = ggml_time_us();
    if (have_mem) fprintf(stderr, "memory: %s\n", memory_describe(mem_start).c_str());
    FILE *trace = nullptr;
    if (!args.trace_routing.empty()) {
        if (!hp.moe()) fprintf(stderr, "warning: --trace-routing ignored for a dense model\n");
        trace = fopen(args.trace_routing.c_str(), "wb");
        if (!trace) fprintf(stderr, "warning: cannot write %s\n", args.trace_routing.c_str());
        engine.set_routing_trace(trace);
    }
    fprintf(stderr, "memory: weights %.2f GiB  kv %.2f GiB  compute %.2f GiB  metal working-set cap %.2f GiB  load %.0f ms\n",
            st.weight_bytes / 1073741824.0, st.kv_bytes / 1073741824.0, st.compute_bytes / 1073741824.0,
            lstk_metal_recommended_working_set() / 1073741824.0, st.load_ms);

    int rc = 0;
    if (!args.compare_ref.empty()) {
        rc = run_compare_ref(args, tok, engine);
    } else if (args.bench_prompt > 0) {
        rc = run_bench(args, engine);
    } else if (args.serve) {
        ServerParams sp;
        sp.host = args.host;
        sp.port = args.port;
        sp.n_ctx = args.n_ctx;
        sp.max_tokens_default = args.n_predict;
        sp.sampling = args.sampling;
        sp.verbose = args.verbose;
        rc = run_server(sp, tok, engine);
    } else if (args.chat) {
        rc = run_chat(args, tok, engine);
    } else {
        // Default mode: single-shot completion. With --parity, decoding is greedy and follows llama.cpp's
        // choice at each step so the two runs stay on the same token sequence.
        std::vector<int32_t> prompt = raw ? args.tokens : tok.encode(args.prompt, true, true);
        if (prompt.empty()) {
            fprintf(stderr, "error: empty prompt\n");
            return 1;
        }
        std::vector<float> logits;
        const int64_t t_prompt0 = ggml_time_us();
        if (!engine.decode(0, prompt.data(), (int) prompt.size(), logits, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        const double prompt_ms = (ggml_time_us() - t_prompt0) / 1000.0;
        if (args.print_logits > 0) print_top(logits, args.print_logits);

        RefModel ref;
        if (args.parity) {
            if (!ref.load(args) || !ref.decode(prompt.data(), (int32_t) prompt.size())) {
                fprintf(stderr, "error: failed to run the llama.cpp reference model\n");
                return 1;
            }
            report("prompt   ", diff_logits(ref.logits(), logits.data(), (int) hp.n_vocab));
        }

        Sampler sampler(args.sampling);
        for (int32_t t : prompt) sampler.accept(t);
        Utf8Printer out;
        if (!raw) printf("%s", args.prompt.c_str());
        fflush(stdout);
        int generated = 0;
        int mismatches = 0;
        const int64_t t_gen0 = ggml_time_us();
        for (int i = 0; i < args.n_predict; i++) {
            int32_t next = args.parity ? argmax(logits.data(), (int) hp.n_vocab) : sampler.sample(logits);
            if (args.parity) {
                Diff d = diff_logits(ref.logits(), logits.data(), (int) hp.n_vocab);
                if (d.ref_arg != d.our_arg) mismatches++;
                if (args.verbose || d.ref_arg != d.our_arg) {
                    printf("\n");
                    char label[32];
                    snprintf(label, sizeof(label), "step %-4d", i);
                    report(label, d);
                }
                next = d.ref_arg;
            }
            sampler.accept(next);
            if (raw) {
                printf("[%d]", next);
            } else {
                if (tok.is_eog(next)) break;
                out.write(tok.piece(next));
            }
            generated++;
            if (!engine.decode(0, &next, 1, logits, err)) {
                fprintf(stderr, "\nerror: %s\n", err.c_str());
                return 1;
            }
            if (args.print_logits > 0) {
                printf("\n");
                print_top(logits, args.print_logits);
            }
            if (args.parity && !ref.decode(&next, 1)) {
                fprintf(stderr, "\nerror: reference decode failed\n");
                return 1;
            }
        }
        out.flush();
        const double gen_ms = (ggml_time_us() - t_gen0) / 1000.0;
        printf("\n");
        fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.1f tok/s)   decode: %d tokens in %.0f ms (%.2f tok/s)\n", prompt.size(),
                prompt_ms, prompt.size() * 1000.0 / prompt_ms, generated, gen_ms, generated * 1000.0 / std::max(gen_ms, 1.0));
        if (args.parity) fprintf(stderr, "parity: %d/%d greedy steps disagreed with llama.cpp\n", mismatches, generated);
    }
    if (trace) fclose(trace);
    MemorySnapshot mem_end;
    if (have_mem && memory_snapshot(mem_end)) {
        fprintf(stderr, "memory pressure during run: %s\n",
                memory_pressure_delta(mem_start, mem_end, (ggml_time_us() - t_run0) / 1e6).c_str());
    }
    const EngineStats &fs = engine.stats();
    if (fs.streaming && fs.experts.acquires > 0) {
        const double loads = (double) (fs.experts.hits + fs.experts.misses);
        fprintf(stderr, "experts: %llu loads, hit rate %.1f%%, %.1f MiB read per acquire-set, I/O wait %.2f ms per segment avg, "
                        "segment overhead %.2f ms avg over %llu segments\n",
                (unsigned long long) (fs.experts.hits + fs.experts.misses), loads > 0 ? 100.0 * fs.experts.hits / loads : 0.0,
                fs.experts.acquires > 0 ? fs.experts.bytes_read / 1048576.0 / fs.experts.acquires : 0.0,
                fs.experts.acquires > 0 ? fs.experts.wait_ms / fs.experts.acquires : 0.0,
                fs.segments > 0 ? fs.segment_overhead_ms / fs.segments : 0.0, (unsigned long long) fs.segments);
    }
    llama_backend_free();
    return rc;
}
