// tokenizer.h - text <-> token ids, and chat-template formatting.
//
// Delegates to llama.cpp's tokenizer, loaded in vocab-only mode (reads the vocabulary from the
// GGUF metadata, no weights). Tokenization is intricate (BPE merges, pre-tokenizer regexes, special
// tokens) and matching llama.cpp exactly is what lets --parity compare logits token for token.

#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Owns a vocab-only llama_model.
struct Tokenizer {
    llama_model *model = nullptr;
    const llama_vocab *vocab = nullptr;

    ~Tokenizer();
    bool load(const std::string &path, std::string &err);
    // add_special: prepend BOS if the model wants one. parse_special: turn text like <|im_start|> into its token.
    std::vector<int32_t> encode(const std::string &text, bool add_special, bool parse_special) const;
    // A token's text. May be a partial UTF-8 character; callers buffer before printing.
    std::string piece(int32_t token) const;
    // End-of-generation token (EOS, <|im_end|>, ...).
    bool is_eog(int32_t token) const;
    int32_t n_vocab() const;
    std::string model_name() const;
    // Formats (role, content) messages with the model's chat template (falls back to ChatML).
    // add_assistant appends the opening of an assistant turn so the model writes the reply.
    std::string apply_template(const std::vector<std::pair<std::string, std::string>> &msgs, bool add_assistant) const;
};
