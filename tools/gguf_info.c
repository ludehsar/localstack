#include "gguf.h"
#include <stdio.h>
#include <string.h>

static void print_kv(const struct gguf_context *g, int64_t i) {
    const char *key = gguf_get_key(g, i);
    enum gguf_type t = gguf_get_kv_type(g, i);
    printf("  %-56s ", key);
    switch (t) {
        case GGUF_TYPE_UINT8:   printf("u8   %u\n", gguf_get_val_u8(g, i)); break;
        case GGUF_TYPE_INT8:    printf("i8   %d\n", gguf_get_val_i8(g, i)); break;
        case GGUF_TYPE_UINT16:  printf("u16  %u\n", gguf_get_val_u16(g, i)); break;
        case GGUF_TYPE_INT16:   printf("i16  %d\n", gguf_get_val_i16(g, i)); break;
        case GGUF_TYPE_UINT32:  printf("u32  %u\n", gguf_get_val_u32(g, i)); break;
        case GGUF_TYPE_INT32:   printf("i32  %d\n", gguf_get_val_i32(g, i)); break;
        case GGUF_TYPE_FLOAT32: printf("f32  %g\n", gguf_get_val_f32(g, i)); break;
        case GGUF_TYPE_BOOL:    printf("bool %d\n", gguf_get_val_bool(g, i)); break;
        case GGUF_TYPE_UINT64:  printf("u64  %llu\n", (unsigned long long)gguf_get_val_u64(g, i)); break;
        case GGUF_TYPE_INT64:   printf("i64  %lld\n", (long long)gguf_get_val_i64(g, i)); break;
        case GGUF_TYPE_FLOAT64: printf("f64  %g\n", gguf_get_val_f64(g, i)); break;
        case GGUF_TYPE_STRING: {
            const char *s = gguf_get_val_str(g, i);
            size_t n = strlen(s);
            printf("str  \"%.*s\"%s\n", (int)(n > 80 ? 80 : n), s, n > 80 ? "..." : "");
            break;
        }
        case GGUF_TYPE_ARRAY:
            printf("arr[%s] n=%zu\n", gguf_type_name(gguf_get_arr_type(g, i)), gguf_get_arr_n(g, i));
            break;
        default: printf("?\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.gguf> [--tensors] [--filter substr]\n", argv[0]);
        return 2;
    }
    int show_tensors = 0;
    const char *filter = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tensors")) show_tensors = 1;
        else if (!strcmp(argv[i], "--filter") && i + 1 < argc) filter = argv[++i];
    }
    struct gguf_init_params p = { 1, NULL };
    struct gguf_context *g = gguf_init_from_file(argv[1], p);
    if (!g) {
        fprintf(stderr, "failed to open %s\n", argv[1]);
        return 1;
    }
    printf("version %u  alignment %zu  data offset %zu  tensors %lld  kv %lld\n", gguf_get_version(g), gguf_get_alignment(g),
           gguf_get_data_offset(g), (long long)gguf_get_n_tensors(g), (long long)gguf_get_n_kv(g));
    printf("[kv]\n");
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        if (filter && !strstr(gguf_get_key(g, i), filter)) continue;
        print_kv(g, i);
    }
    if (show_tensors) {
        printf("[tensors]\n");
        size_t total = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char *name = gguf_get_tensor_name(g, i);
            if (filter && !strstr(name, filter)) continue;
            const int64_t *ne = gguf_get_tensor_ne(g, i);
            size_t sz = gguf_get_tensor_size(g, i);
            total += sz;
            printf("  %-40s %-8s [%lld, %lld, %lld, %lld] %12zu bytes @ %zu\n", name, ggml_type_name(gguf_get_tensor_type(g, i)),
                   (long long)ne[0], (long long)ne[1], (long long)ne[2], (long long)ne[3], sz, gguf_get_tensor_offset(g, i));
        }
        printf("  total %.3f GiB\n", total / 1073741824.0);
    }
    gguf_free(g);
    return 0;
}
