// tokenizer.cpp - thin wrappers over llama.cpp's vocabulary and chat-template API.
// llama.cpp functions return a negative count when the output buffer is too small; each wrapper
// then resizes to that size and retries.

#include "tokenizer.h"

Tokenizer::~Tokenizer() {
    if (model) llama_model_free(model);
}

bool Tokenizer::load(const std::string &path, std::string &err) {
    llama_model_params mp = llama_model_default_params();
    // Load only the tokenizer metadata, not the multi-GB weights (the Engine loads those itself).
    mp.vocab_only = true;
    model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        err = "failed to load vocabulary from " + path;
        return false;
    }
    vocab = llama_model_get_vocab(model);
    return vocab != nullptr;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, bool add_special, bool parse_special) const {
    std::vector<int32_t> out(text.size() + 8);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special,
                               parse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special,
                           parse_special);
    }
    out.resize(n > 0 ? (size_t) n : 0);
    return out;
}

std::string Tokenizer::piece(int32_t token) const {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::string big((size_t) -n, '\0');
        n = llama_token_to_piece(vocab, token, big.data(), (int32_t) big.size(), 0, true);
        big.resize(n > 0 ? (size_t) n : 0);
        return big;
    }
    return std::string(buf, (size_t) n);
}

bool Tokenizer::is_eog(int32_t token) const {
    return llama_vocab_is_eog(vocab, token);
}

int32_t Tokenizer::n_vocab() const {
    return llama_vocab_n_tokens(vocab);
}

std::string Tokenizer::model_name() const {
    char buf[256];
    if (llama_model_meta_val_str(model, "general.name", buf, sizeof(buf)) > 0) return buf;
    return "model";
}

std::string Tokenizer::apply_template(const std::vector<std::pair<std::string, std::string>> &msgs, bool add_assistant) const {
    // Use the template stored in the GGUF; fall back to ChatML (the format Qwen uses) if it is missing or unsupported.
    const char *tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) tmpl = "chatml";
    std::vector<llama_chat_message> cm;
    for (const auto &m : msgs) cm.push_back({ m.first.c_str(), m.second.c_str() });
    std::string buf(4096, '\0');
    int32_t n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_assistant, buf.data(), (int32_t) buf.size());
    if (n < 0) {
        tmpl = "chatml";
        n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_assistant, buf.data(), (int32_t) buf.size());
        if (n < 0) return "";
    }
    if (n > (int32_t) buf.size()) {
        buf.resize((size_t) n);
        n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_assistant, buf.data(), (int32_t) buf.size());
    }
    buf.resize(n > 0 ? (size_t) n : 0);
    return buf;
}
