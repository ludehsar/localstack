#pragma once

#include "engine.h"
#include "sampler.h"
#include "tokenizer.h"

#include <string>

struct ServerParams {
    std::string host = "127.0.0.1";
    int port = 8080;
    uint32_t n_ctx = 4096;
    int max_tokens_default = 512;
    SamplerParams sampling;
    bool verbose = false;
};

int run_server(const ServerParams &params, Tokenizer &tok, Engine &engine);
