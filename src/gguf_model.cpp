#include "gguf_model.h"

#include <fcntl.h>
#include <unistd.h>

#include <vector>

GgufModel::~GgufModel() {
    close();
}

bool GgufModel::open(const std::string &file, std::string &err) {
    close();
    path = file;
    fd = ::open(file.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "cannot open " + file;
        return false;
    }
    gguf_init_params params = { true, &meta };
    gguf = gguf_init_from_file(file.c_str(), params);
    if (!gguf) {
        err = "not a readable GGUF file: " + file;
        return false;
    }
    data_offset = gguf_get_data_offset(gguf);
    arch = get_str("general.architecture");
    if (arch.empty()) {
        err = "GGUF has no general.architecture";
        return false;
    }
    return true;
}

void GgufModel::close() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (wctx) ggml_free(wctx);
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
    if (fd >= 0) ::close(fd);
    buffer = nullptr;
    wctx = nullptr;
    meta = nullptr;
    gguf = nullptr;
    fd = -1;
    loaded.clear();
}

bool GgufModel::has(const std::string &key) const {
    return gguf && gguf_find_key(gguf, key.c_str()) >= 0;
}

int64_t GgufModel::get_int(const std::string &key, int64_t def) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(gguf, id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(gguf, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf, id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(gguf, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(gguf, id);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(gguf, id);
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(gguf, id);
        default: return def;
    }
}

double GgufModel::get_float(const std::string &key, double def) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(gguf, id);
        default: return (double) get_int(key, (int64_t) def);
    }
}

std::string GgufModel::get_str(const std::string &key, const std::string &def) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf, id);
}

ggml_tensor *GgufModel::tensor(const std::string &name) const {
    auto it = loaded.find(name);
    return it == loaded.end() ? nullptr : it->second;
}

bool GgufModel::info(const std::string &name, GgufTensorInfo &out) const {
    int64_t id = gguf_find_tensor(gguf, name.c_str());
    if (id < 0) return false;
    out.type = gguf_get_tensor_type(gguf, id);
    const int64_t *ne = gguf_get_tensor_ne(gguf, id);
    for (int i = 0; i < 4; i++) out.ne[i] = ne[i];
    out.file_offset = data_offset + gguf_get_tensor_offset(gguf, id);
    out.nbytes = gguf_get_tensor_size(gguf, id);
    return true;
}

size_t GgufModel::total_bytes() const {
    size_t total = 0;
    for (ggml_tensor *t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) total += ggml_nbytes(t);
    return total;
}

size_t GgufModel::loaded_bytes() const {
    size_t total = 0;
    for (const auto &kv : loaded) total += ggml_nbytes(kv.second);
    return total;
}

static bool read_exact(int fd, void *dst, size_t size, off_t offset) {
    uint8_t *p = (uint8_t *) dst;
    while (size > 0) {
        ssize_t n = pread(fd, p, size, offset);
        if (n <= 0) return false;
        p += n;
        size -= (size_t) n;
        offset += n;
    }
    return true;
}

bool GgufModel::load_all(ggml_backend_t backend, std::string &err, const std::function<bool(const std::string &)> &skip) {
    const int64_t n = gguf_get_n_tensors(gguf);
    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) (n + 1), nullptr, true };
    wctx = ggml_init(wp);
    for (ggml_tensor *t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        if (skip && skip(t->name)) continue;
        ggml_tensor *d = ggml_dup_tensor(wctx, t);
        ggml_set_name(d, t->name);
        loaded[t->name] = d;
    }
    buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!buffer) {
        err = "failed to allocate the weight buffer";
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> staging;
    for (auto &kv : loaded) {
        ggml_tensor *t = kv.second;
        int64_t id = gguf_find_tensor(gguf, t->name);
        if (id < 0) {
            err = std::string("tensor missing from GGUF index: ") + t->name;
            return false;
        }
        const size_t nbytes = ggml_nbytes(t);
        const off_t off = (off_t) (data_offset + gguf_get_tensor_offset(gguf, id));
        if (ggml_backend_buffer_is_host(t->buffer)) {
            if (!read_exact(fd, t->data, nbytes, off)) {
                err = std::string("short read for tensor ") + t->name;
                return false;
            }
        } else {
            staging.resize(nbytes);
            if (!read_exact(fd, staging.data(), nbytes, off)) {
                err = std::string("short read for tensor ") + t->name;
                return false;
            }
            ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        }
    }
    return true;
}
