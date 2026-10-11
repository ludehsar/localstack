#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct SamplerParams {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    uint32_t seed = 0;
};

struct Sampler {
    SamplerParams params;
    std::mt19937 rng;
    std::vector<int32_t> recent;

    explicit Sampler(const SamplerParams &p);
    int32_t sample(std::vector<float> &logits);
    void accept(int32_t token);
    void reset();
};
