// server.cpp - OpenAI-compatible chat completions over HTTP (cpp-httplib + nlohmann/json, both vendored
// in llama.cpp).
//
// Concurrency: one generation at a time. A Generation holds the shared mutex for its whole life,
// including across streaming callbacks, so other requests wait. All requests share KV sequence 0;
// `Shared::fed` mirrors the tokens in it, so a new request whose prompt starts with the same tokens
// (a multi-turn chat resending its history) only processes the new suffix.

#include "server.h"

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>

using json = nlohmann::ordered_json;

namespace {

// State shared by all requests: the engine, the mutex that serializes them, and the cached token prefix.
struct Shared {
    Tokenizer &tok;
    Engine &engine;
    const ServerParams &params;
    std::mutex mutex;
    // Exactly the tokens currently held in KV sequence 0.
    std::vector<int32_t> fed;
    std::atomic<uint64_t> counter{ 0 };
    std::string model_name;
    Shared(Tokenizer &t, Engine &e, const ServerParams &p) : tok(t), engine(e), params(p), model_name(t.model_name()) {}
};

// Length of `s` without a trailing incomplete UTF-8 character, so streamed chunks never split a character.
size_t utf8_safe_len(const std::string &s) {
    for (size_t back = 1; back <= 4 && back <= s.size(); back++) {
        const unsigned char c = (unsigned char) s[s.size() - back];
        if ((c & 0xC0) == 0x80) continue;
        const size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        return back < need ? s.size() - back : s.size();
    }
    return s.size();
}

// One in-flight completion: holds the lock, sampler, current logits, and the reply text generated so far.
struct Generation {
    std::shared_ptr<Shared> sh;
    std::unique_lock<std::mutex> lock;
    Sampler sampler;
    std::vector<float> logits;
    std::string id;
    int64_t created = 0;
    int max_tokens = 0;
    std::vector<std::string> stops;
    std::string reply;
    // Bytes of `reply` already sent to the client.
    size_t emitted = 0;
    int generated = 0;
    int prompt_tokens = 0;
    int reused_tokens = 0;
    bool finished = false;
    std::string finish_reason = "stop";
    int64_t t_start = 0;
    double prompt_ms = 0.0;

    Generation(std::shared_ptr<Shared> s, const SamplerParams &sp) : sh(std::move(s)), lock(sh->mutex), sampler(sp) {}

    // Length of the longest tail of `reply` that is a prefix of some stop string. That text is held back
    // from streaming because, if the stop string completes, it must not appear in the output.
    size_t held_suffix() const {
        size_t held = 0;
        for (const std::string &stop : stops) {
            for (size_t n = std::min(stop.size() - 1, reply.size()); n > held; n--) {
                if (reply.compare(reply.size() - n, n, stop, 0, n) == 0) {
                    held = n;
                    break;
                }
            }
        }
        return held;
    }

    // Finds a complete stop string, searching only near the unsent tail (text before it was already checked).
    bool stop_hit(size_t &cut) const {
        size_t from = emitted > 64 ? emitted - 64 : 0;
        for (const std::string &stop : stops) {
            if (stop.empty()) continue;
            size_t pos = reply.find(stop, from);
            if (pos != std::string::npos) {
                cut = pos;
                return true;
            }
        }
        return false;
    }

    // Returns the next chunk of `reply` that is safe to send (everything, if final).
    std::string drain(bool final) {
        size_t limit = final ? reply.size() : std::min(utf8_safe_len(reply), reply.size() - held_suffix());
        if (limit < emitted) limit = emitted;
        std::string out = reply.substr(emitted, limit - emitted);
        emitted = limit;
        return out;
    }

    // Samples one token and feeds it to the engine. Sets finished on EOG, a stop string, or max_tokens.
    // `delta` gets the newly sendable text.
    bool step(std::string &delta, std::string &err) {
        delta.clear();
        if (finished) return true;
        Engine &engine = sh->engine;
        const int32_t next = sampler.sample(logits);
        sampler.accept(next);
        sh->fed.push_back(next);
        if (sh->tok.is_eog(next)) {
            // Feed the EOG token into the KV too, so the next request's templated history lines up with `fed`.
            engine.decode(0, &next, 1, logits, err);
            finished = true;
            finish_reason = "stop";
            delta = drain(true);
            return true;
        }
        reply += sh->tok.piece(next);
        generated++;
        size_t cut = 0;
        if (stop_hit(cut)) {
            reply.resize(cut);
            finished = true;
            finish_reason = "stop";
            delta = drain(true);
            return true;
        }
        if (generated >= max_tokens) {
            finished = true;
            finish_reason = "length";
            delta = drain(true);
            return true;
        }
        if (!engine.decode(0, &next, 1, logits, err)) {
            finished = true;
            finish_reason = "length";
            delta = drain(true);
            return false;
        }
        delta = drain(false);
        return true;
    }
};

// OpenAI message content is either a string or an array of parts; keep only the text parts.
std::string content_text(const json &content) {
    if (content.is_string()) return content.get<std::string>();
    std::string out;
    if (content.is_array()) {
        for (const json &part : content) {
            if (part.is_object() && part.value("type", "") == "text") out += part.value("text", "");
        }
    }
    return out;
}

// Validates the request, applies per-request sampling overrides, templates and tokenizes the messages,
// reuses the cached KV prefix, and runs the prompt so `g.logits` is ready for the first sample.
bool prepare(Generation &g, const json &body, std::string &error) {
    Shared &sh = *g.sh;
    std::vector<std::pair<std::string, std::string>> msgs;
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        error = "messages must be a non-empty array";
        return false;
    }
    for (const json &m : body["messages"]) {
        if (!m.is_object() || !m.contains("role")) {
            error = "each message needs a role";
            return false;
        }
        msgs.push_back({ m["role"].get<std::string>(), content_text(m.value("content", json())) });
    }
    SamplerParams sp = sh.params.sampling;
    if (body.contains("temperature")) sp.temperature = body["temperature"].get<float>();
    if (body.contains("top_p")) sp.top_p = body["top_p"].get<float>();
    if (body.contains("top_k")) sp.top_k = body["top_k"].get<int>();
    if (body.contains("min_p")) sp.min_p = body["min_p"].get<float>();
    if (body.contains("seed")) sp.seed = body["seed"].get<uint32_t>();
    if (body.contains("repeat_penalty")) sp.repeat_penalty = body["repeat_penalty"].get<float>();
    g.sampler = Sampler(sp);
    g.max_tokens = body.value("max_tokens", body.value("max_completion_tokens", sh.params.max_tokens_default));
    if (body.contains("stop")) {
        if (body["stop"].is_string()) g.stops.push_back(body["stop"].get<std::string>());
        else if (body["stop"].is_array()) {
            for (const json &s : body["stop"]) {
                if (s.is_string()) g.stops.push_back(s.get<std::string>());
            }
        }
    }

    const std::string formatted = sh.tok.apply_template(msgs, true);
    std::vector<int32_t> tokens = sh.tok.encode(formatted, true, true);
    if (tokens.empty()) {
        error = "prompt tokenized to nothing";
        return false;
    }
    if (tokens.size() + 1 >= sh.params.n_ctx) {
        error = "prompt is longer than the context window";
        return false;
    }
    if ((int) (sh.params.n_ctx - tokens.size() - 1) < g.max_tokens) g.max_tokens = (int) (sh.params.n_ctx - tokens.size() - 1);

    // KV prefix reuse: keep the longest common prefix, drop the rest. If the whole prompt is already cached,
    // back off one token: decoding it again is what produces the logits to sample from.
    size_t common = 0;
    while (common < sh.fed.size() && common < tokens.size() && sh.fed[common] == tokens[common]) common++;
    if (common == tokens.size()) common = tokens.size() - 1;
    sh.engine.truncate_seq(0, (int32_t) common);
    sh.fed.resize(common);
    g.reused_tokens = (int) common;
    g.prompt_tokens = (int) tokens.size();

    const int64_t t0 = ggml_time_us();
    std::string err;
    if (!sh.engine.decode(0, tokens.data() + common, (int) (tokens.size() - common), g.logits, err)) {
        sh.engine.remove_seq(0);
        sh.fed.clear();
        error = err;
        return false;
    }
    g.prompt_ms = (ggml_time_us() - t0) / 1000.0;
    sh.fed.insert(sh.fed.end(), tokens.begin() + (long) common, tokens.end());
    for (int32_t t : tokens) g.sampler.accept(t);
    g.t_start = ggml_time_us();
    g.id = "chatcmpl-" + std::to_string(++sh.counter);
    g.created = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return true;
}

json usage_json(const Generation &g) {
    return json{ { "prompt_tokens", g.prompt_tokens }, { "completion_tokens", g.generated }, { "total_tokens", g.prompt_tokens + g.generated } };
}

json chunk_json(const Generation &g, const json &delta, const json &finish) {
    return json{ { "id", g.id },
                 { "object", "chat.completion.chunk" },
                 { "created", g.created },
                 { "model", g.sh->model_name },
                 { "choices", json::array({ json{ { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
}

void log_request(const Generation &g) {
    const double gen_ms = (ggml_time_us() - g.t_start) / 1000.0;
    fprintf(stderr, "[%s] prompt %d tokens (%d reused) in %.0f ms, generated %d in %.0f ms (%.1f tok/s), finish=%s\n", g.id.c_str(),
            g.prompt_tokens, g.reused_tokens, g.prompt_ms, g.generated, gen_ms, g.generated * 1000.0 / std::max(gen_ms, 1.0),
            g.finish_reason.c_str());
}

void error_response(httplib::Response &res, int status, const std::string &message) {
    res.status = status;
    res.set_content(json{ { "error", json{ { "message", message }, { "type", "invalid_request_error" } } } }.dump(), "application/json");
}

} // namespace

// Routes: CORS preflight, /health, /v1/models, /v1/chat/completions.
int run_server(const ServerParams &params, Tokenizer &tok, Engine &engine) {
    auto sh = std::make_shared<Shared>(tok, engine, params);
    httplib::Server svr;

    svr.set_default_headers({ { "Access-Control-Allow-Origin", "*" },
                              { "Access-Control-Allow-Headers", "Content-Type, Authorization" },
                              { "Access-Control-Allow-Methods", "GET, POST, OPTIONS" } });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response &res) { res.status = 204; });

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(json{ { "status", "ok" } }.dump(), "application/json");
    });

    svr.Get("/v1/models", [sh](const httplib::Request &, httplib::Response &res) {
        json data = json::array({ json{ { "id", sh->model_name }, { "object", "model" }, { "owned_by", "localstack" } } });
        res.set_content(json{ { "object", "list" }, { "data", data } }.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [sh](const httplib::Request &req, httplib::Response &res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            error_response(res, 400, "invalid JSON body");
            return;
        }
        auto g = std::make_shared<Generation>(sh, sh->params.sampling);
        std::string error;
        if (!prepare(*g, body, error)) {
            error_response(res, 400, error);
            return;
        }
        const bool stream = body.value("stream", false);
        if (!stream) {
            std::string delta, err;
            while (!g->finished) {
                if (!g->step(delta, err) && !err.empty()) break;
            }
            log_request(*g);
            json out{ { "id", g->id },
                      { "object", "chat.completion" },
                      { "created", g->created },
                      { "model", sh->model_name },
                      { "choices", json::array({ json{ { "index", 0 },
                                                       { "message", json{ { "role", "assistant" }, { "content", g->reply } } },
                                                       { "finish_reason", g->finish_reason } } }) },
                      { "usage", usage_json(*g) } };
            res.set_content(out.dump(), "application/json");
            return;
        }
        // Streaming: httplib calls this provider repeatedly; each call generates one token and sends its text as an
        // SSE "data:" chunk. The first chunk announces the assistant role; the last carries finish_reason and usage.
        auto first = std::make_shared<bool>(true);
        res.set_chunked_content_provider("text/event-stream", [g, first](size_t, httplib::DataSink &sink) {
            auto send = [&](const json &j) {
                const std::string s = "data: " + j.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            if (*first) {
                *first = false;
                if (!send(chunk_json(*g, json{ { "role", "assistant" }, { "content", "" } }, nullptr))) return false;
            }
            std::string delta, err;
            g->step(delta, err);
            if (!delta.empty() && !send(chunk_json(*g, json{ { "content", delta } }, nullptr))) return false;
            if (g->finished) {
                json last = chunk_json(*g, json::object(), g->finish_reason);
                last["usage"] = usage_json(*g);
                if (!send(last)) return false;
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                log_request(*g);
                sink.done();
            }
            return true;
        });
    });

    fprintf(stderr, "serving %s on http://%s:%d (POST /v1/chat/completions, GET /v1/models, GET /health)\n", sh->model_name.c_str(),
            params.host.c_str(), params.port);
    if (!svr.listen(params.host, params.port)) {
        fprintf(stderr, "error: cannot listen on %s:%d\n", params.host.c_str(), params.port);
        return 1;
    }
    return 0;
}
