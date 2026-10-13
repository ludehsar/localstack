// sampler.h - turns the model's logits into the next token.
//
// The model outputs one score per vocabulary entry. Greedy decoding takes the highest; sampling
// draws from a softmax that temperature, top-k, min-p and top-p reshape and truncate so output is
// varied but stays on likely tokens. The order of these steps is recorded in docs/decisions.md.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

// Defaults match common llama.cpp settings. temperature <= 0 means greedy.
struct SamplerParams {
    float temperature = 0.7f;
    // Keep only the k highest-scoring tokens (0 = off).
    int top_k = 40;
    // Keep the smallest set of tokens whose probabilities sum to top_p (1 = off).
    float top_p = 0.95f;
    // Drop tokens less likely than min_p times the most likely token (0 = off).
    float min_p = 0.05f;
    // Penalize tokens among the last repeat_last_n accepted ones (1 = off).
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    // RNG seed; 0 = random.
    uint32_t seed = 0;
};

// Per-generation sampler state: parameters, RNG, and the recent-token window for the repeat penalty.
struct Sampler {
    SamplerParams params;
    std::mt19937 rng;
    std::vector<int32_t> recent;

    explicit Sampler(const SamplerParams &p);
    // Picks a token. Note: applies the repeat penalty to `logits` in place.
    int32_t sample(std::vector<float> &logits);
    // Records a token (prompt or generated) in the repeat-penalty window.
    void accept(int32_t token);
    void reset();
};
