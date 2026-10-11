#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct Tokenizer {
    llama_model *model = nullptr;
    const llama_vocab *vocab = nullptr;

    ~Tokenizer();
    bool load(const std::string &path, std::string &err);
    std::vector<int32_t> encode(const std::string &text, bool add_special, bool parse_special) const;
    std::string piece(int32_t token) const;
    bool is_eog(int32_t token) const;
    int32_t n_vocab() const;
    std::string model_name() const;
    std::string apply_template(const std::vector<std::pair<std::string, std::string>> &msgs, bool add_assistant) const;
};
