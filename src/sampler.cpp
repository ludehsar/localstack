#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

Sampler::Sampler(const SamplerParams &p) : params(p), rng(p.seed ? p.seed : std::random_device{}()) {}

void Sampler::reset() {
    recent.clear();
}

void Sampler::accept(int32_t token) {
    recent.push_back(token);
    if (params.repeat_last_n > 0 && (int) recent.size() > params.repeat_last_n) recent.erase(recent.begin());
}

int32_t Sampler::sample(std::vector<float> &logits) {
    const int n = (int) logits.size();
    if (params.repeat_penalty != 1.0f) {
        for (int32_t t : recent) {
            float &l = logits[t];
            l = l > 0 ? l / params.repeat_penalty : l * params.repeat_penalty;
        }
    }
    if (params.temperature <= 0.0f) return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());

    std::vector<int32_t> cand(n);
    std::iota(cand.begin(), cand.end(), 0);
    auto greater = [&](int32_t a, int32_t b) { return logits[a] > logits[b]; };
    int keep = n;
    if (params.top_k > 0 && params.top_k < n) {
        keep = params.top_k;
        std::nth_element(cand.begin(), cand.begin() + keep, cand.end(), greater);
    }
    cand.resize(keep);
    std::sort(cand.begin(), cand.end(), greater);

    std::vector<float> probs(keep);
    const float max_logit = logits[cand[0]];
    double sum = 0.0;
    for (int i = 0; i < keep; i++) {
        probs[i] = expf((logits[cand[i]] - max_logit) / params.temperature);
        sum += probs[i];
    }
    for (float &p : probs) p = (float) (p / sum);

    int cut = keep;
    if (params.min_p > 0.0f) {
        const float floor = params.min_p * probs[0];
        while (cut > 1 && probs[cut - 1] < floor) cut--;
    }
    if (params.top_p < 1.0f) {
        double acc = 0.0;
        int i = 0;
        for (; i < cut; i++) {
            acc += probs[i];
            if (acc >= params.top_p) {
                i++;
                break;
            }
        }
        cut = std::max(1, std::min(cut, i));
    }
    double total = 0.0;
    for (int i = 0; i < cut; i++) total += probs[i];
    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);
    for (int i = 0; i < cut; i++) {
        r -= probs[i];
        if (r <= 0.0) return cand[i];
    }
    return cand[cut - 1];
}
