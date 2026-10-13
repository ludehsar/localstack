// gguf_model.h - reads a GGUF model file and loads its tensors into a ggml backend buffer.
//
// GGUF (llama.cpp's model format) is one file holding a key/value metadata section (architecture,
// layer count, head counts, ...), a tensor index (name, type, shape, offset), and the raw tensor
// data. GgufModel exposes the metadata getters, the tensor index, and load_all(), which copies
// tensors into GPU-visible memory. Streaming mode skips the expert tensors and instead reads them
// on demand through the raw file descriptor `fd`, using offsets from info().

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

// A tensor's type, shape, and byte range in the file, without loading it.
struct GgufTensorInfo {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = { 1, 1, 1, 1 };
    size_t file_offset = 0;
    size_t nbytes = 0;
    // For a stacked expert tensor [.., .., n_expert], experts are the outermost dimension, so each
    // expert is one contiguous run of nbytes / n_expert bytes. This is what makes per-expert pread possible.
    size_t expert_stride() const { return ne[2] > 0 ? nbytes / (size_t) ne[2] : 0; }
};

// An open GGUF file. Call open(), then load_all(); tensor() returns the loaded copies by name.
struct GgufModel {
    std::string path;
    std::string arch;
    int fd = -1;
    // Parsed header + metadata (llama.cpp's gguf API).
    gguf_context *gguf = nullptr;
    // Tensor descriptors from the index, with no data attached.
    ggml_context *meta = nullptr;
    // The loaded weight tensors and the backend buffer that stores them.
    ggml_context *wctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    // File offset where tensor data begins; tensor offsets are relative to it.
    size_t data_offset = 0;
    // Name -> loaded tensor, for every tensor load_all() did not skip.
    std::unordered_map<std::string, ggml_tensor *> loaded;

    ~GgufModel();
    bool open(const std::string &file, std::string &err);
    void close();

    // Metadata lookups; numeric getters accept any integer/float GGUF type and return `def` if absent.
    bool has(const std::string &key) const;
    int64_t get_int(const std::string &key, int64_t def = 0) const;
    double get_float(const std::string &key, double def = 0.0) const;
    std::string get_str(const std::string &key, const std::string &def = "") const;

    ggml_tensor *tensor(const std::string &name) const;
    bool info(const std::string &name, GgufTensorInfo &out) const;
    // Size of all tensors in the file / of the tensors actually loaded.
    size_t total_bytes() const;
    size_t loaded_bytes() const;
    // Allocates one buffer on `backend` and reads every tensor into it, except names for which skip() is true.
    bool load_all(ggml_backend_t backend, std::string &err, const std::function<bool(const std::string &)> &skip = nullptr);
};
