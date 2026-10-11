#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

struct GgufTensorInfo {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = { 1, 1, 1, 1 };
    size_t file_offset = 0;
    size_t nbytes = 0;
    size_t expert_stride() const { return ne[2] > 0 ? nbytes / (size_t) ne[2] : 0; }
};

struct GgufModel {
    std::string path;
    std::string arch;
    int fd = -1;
    gguf_context *gguf = nullptr;
    ggml_context *meta = nullptr;
    ggml_context *wctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t data_offset = 0;
    std::unordered_map<std::string, ggml_tensor *> loaded;

    ~GgufModel();
    bool open(const std::string &file, std::string &err);
    void close();

    bool has(const std::string &key) const;
    int64_t get_int(const std::string &key, int64_t def = 0) const;
    double get_float(const std::string &key, double def = 0.0) const;
    std::string get_str(const std::string &key, const std::string &def = "") const;

    ggml_tensor *tensor(const std::string &name) const;
    bool info(const std::string &name, GgufTensorInfo &out) const;
    size_t total_bytes() const;
    size_t loaded_bytes() const;
    bool load_all(ggml_backend_t backend, std::string &err, const std::function<bool(const std::string &)> &skip = nullptr);
};
