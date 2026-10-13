// server.h - an OpenAI-compatible HTTP server on top of the Engine (lstk --serve).
//
// Implements POST /v1/chat/completions (plain JSON or server-sent-event streaming), GET /v1/models
// and GET /health, so existing chat UIs and OpenAI client libraries can talk to the local model.

#pragma once

#include "engine.h"
#include "sampler.h"
#include "tokenizer.h"

#include <string>

// Listen address and request defaults (a request's JSON body can override the sampling fields).
struct ServerParams {
    std::string host = "127.0.0.1";
    int port = 8080;
    uint32_t n_ctx = 4096;
    int max_tokens_default = 512;
    SamplerParams sampling;
    bool verbose = false;
};

// Blocks serving requests until the process exits. Returns non-zero if the port cannot be bound.
int run_server(const ServerParams &params, Tokenizer &tok, Engine &engine);
