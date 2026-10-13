// ssd_bench.c - measure SSD read throughput the way expert streaming will read.
//
// Expert misses become random positioned reads of a few MiB each, so this measures:
//   cold random reads   by chunk size (1-16 MiB) x thread count (1-16), bypassing the page cache
//   cold sequential     16 MiB chunks, one thread, for comparison
//   warm single reads   from the page cache, 16 KiB- vs 2 MiB-aligned destination buffers
//   warm parallel reads from the page cache, by thread count
// Results drove the I/O thread count, buffer alignment, and the --miss-ms cost in tools/cache_sim.py
// (see docs/baseline.md). Use a large file (e.g. a GGUF) that has not been read recently.
//
//   cc -O2 tools/ssd_bench.c -o build/tools/ssd_bench
//   build/tools/ssd_bench FILE [MiB_per_config=256] [warm_iters=64]

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// Per-thread state: threads claim offsets from a shared atomic counter until all are read.
typedef struct {
    int fd;
    size_t chunk;
    size_t align;
    const off_t *offsets;
    size_t n_offsets;
    atomic_size_t *next;
    size_t bytes;
    double max_ms;
} Worker;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *aligned_buf(size_t align, size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

static void *worker_main(void *arg) {
    Worker *w = arg;
    void *buf = aligned_buf(w->align, w->chunk);
    if (!buf) return NULL;
    for (;;) {
        size_t i = atomic_fetch_add(w->next, 1);
        if (i >= w->n_offsets) break;
        double t0 = now_s();
        ssize_t n = pread(w->fd, buf, w->chunk, w->offsets[i]);
        double ms = (now_s() - t0) * 1e3;
        if (n > 0) w->bytes += (size_t)n;
        if (ms > w->max_ms) w->max_ms = ms;
    }
    free(buf);
    return NULL;
}

// xorshift64* PRNG: fast, deterministic offsets without libc rand().
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

// True if no page of the chunk is in the page cache (mincore on an mmap of the file), so a read
// truly hits the SSD. Reads through fd_cold use F_NOCACHE, so they do not warm the cache either.
static int chunk_is_cold(const char *map, size_t file_size, off_t off, size_t chunk, size_t page) {
    size_t len = chunk;
    if ((size_t)off + len > file_size) len = file_size - (size_t)off;
    size_t pages = (len + page - 1) / page;
    char vec[pages];
    if (mincore(map + off, len, vec) != 0) return 1;
    for (size_t i = 0; i < pages; i++) {
        if (vec[i] & MINCORE_INCORE) return 0;
    }
    return 1;
}

// Random page-aligned offsets whose chunks are cold, up to `want` of them.
static size_t pick_cold_offsets(const char *map, size_t file_size, size_t chunk, size_t page, off_t *out, size_t want) {
    size_t got = 0, tries = 0;
    size_t slots = (file_size - chunk) / page;
    while (got < want && tries < want * 50) {
        tries++;
        off_t off = (off_t)((rng_next() % slots) * page);
        if (chunk_is_cold(map, file_size, off, chunk, page)) out[got++] = off;
    }
    return got;
}

// Reads all offsets with `threads` threads; returns wall seconds, total bytes, and the slowest single read.
static double run_parallel(int fd, size_t chunk, size_t align, int threads, const off_t *offsets, size_t n,
                           size_t *bytes_out, double *max_ms_out) {
    pthread_t tids[threads];
    Worker ws[threads];
    atomic_size_t next = 0;
    double t0 = now_s();
    for (int t = 0; t < threads; t++) {
        ws[t] = (Worker){fd, chunk, align, offsets, n, &next, 0, 0.0};
        pthread_create(&tids[t], NULL, worker_main, &ws[t]);
    }
    size_t bytes = 0;
    double max_ms = 0;
    for (int t = 0; t < threads; t++) {
        pthread_join(tids[t], NULL);
        bytes += ws[t].bytes;
        if (ws[t].max_ms > max_ms) max_ms = ws[t].max_ms;
    }
    *bytes_out = bytes;
    *max_ms_out = max_ms;
    return now_s() - t0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [total_mb_per_config=256] [warm_iters=64]\n", argv[0]);
        return 2;
    }
    size_t total_per_cfg = (argc > 2 ? strtoull(argv[2], NULL, 10) : 256) << 20;
    int warm_iters = argc > 3 ? atoi(argv[3]) : 64;
    size_t page = (size_t)getpagesize();
    int fd_cold = open(argv[1], O_RDONLY);
    int fd_warm = open(argv[1], O_RDONLY);
    if (fd_cold < 0 || fd_warm < 0) {
        perror("open");
        return 1;
    }
    struct stat st;
    fstat(fd_cold, &st);
    size_t file_size = (size_t)st.st_size;
    // Two descriptors: fd_cold bypasses the unified buffer cache; fd_warm (and the mmap) go through it.
    if (fcntl(fd_cold, F_NOCACHE, 1) != 0) perror("F_NOCACHE");
    char *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd_warm, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    printf("file: %s\nsize: %.2f GiB  page: %zu  bytes per config: %zu MiB\n", argv[1], file_size / 1073741824.0, page,
           total_per_cfg >> 20);

    size_t chunks_mb[] = {1, 2, 4, 8, 16};
    int threads[] = {1, 2, 4, 8, 16};

    printf("\n[cold random reads] F_NOCACHE, 2 MiB aligned buffers, offsets verified non-resident with mincore\n");
    printf("%8s %8s %10s %8s %10s %10s\n", "chunkMiB", "threads", "GB/s", "reads", "avg_ms", "max_ms");
    for (size_t ci = 0; ci < 5; ci++) {
        size_t chunk = chunks_mb[ci] << 20;
        size_t want = total_per_cfg / chunk;
        if (want < 4) want = 4;
        off_t *offs = malloc(want * sizeof(off_t));
        for (size_t ti = 0; ti < 5; ti++) {
            size_t got = pick_cold_offsets(map, file_size, chunk, page, offs, want);
            if (got == 0) {
                printf("%8zu %8d %10s\n", chunks_mb[ci], threads[ti], "no cold");
                continue;
            }
            size_t bytes;
            double max_ms;
            double dt = run_parallel(fd_cold, chunk, 2u << 20, threads[ti], offs, got, &bytes, &max_ms);
            printf("%8zu %8d %10.2f %8zu %10.3f %10.2f\n", chunks_mb[ci], threads[ti], bytes / dt / 1e9, got,
                   dt * 1e3 / got * threads[ti], max_ms);
        }
        free(offs);
    }

    printf("\n[cold sequential read] 1 thread, 16 MiB chunks, F_NOCACHE\n");
    {
        size_t chunk = 16u << 20;
        size_t want = total_per_cfg / chunk;
        if (want < 4) want = 4;
        off_t *offs = malloc(want * sizeof(off_t));
        size_t slots = (file_size - want * chunk) / page;
        off_t base = (off_t)((rng_next() % slots) * page);
        size_t got = 0;
        for (size_t i = 0; i < want; i++) {
            off_t off = base + (off_t)(i * chunk);
            if (chunk_is_cold(map, file_size, off, chunk, page)) offs[got++] = off;
        }
        size_t bytes;
        double max_ms;
        double dt = run_parallel(fd_cold, chunk, 2u << 20, 1, offs, got, &bytes, &max_ms);
        printf("%.2f GB/s over %zu reads (%zu skipped as resident)\n", bytes / dt / 1e9, got, want - got);
        free(offs);
    }

    printf("\n[warm single-thread reads] one 4 MiB chunk re-read %d times from page cache, by destination alignment\n",
           warm_iters);
    {
        size_t chunk = 4u << 20;
        off_t off = (off_t)((rng_next() % ((file_size - chunk) / page)) * page);
        size_t aligns[] = {16384, 2u << 20};
        for (int ai = 0; ai < 2; ai++) {
            void *buf = aligned_buf(aligns[ai], chunk);
            pread(fd_warm, buf, chunk, off);
            double t0 = now_s();
            for (int i = 0; i < warm_iters; i++) pread(fd_warm, buf, chunk, off);
            double dt = now_s() - t0;
            printf("align %8zu: %8.2f GB/s  %.3f ms per read\n", aligns[ai], (double)chunk * warm_iters / dt / 1e9,
                   dt * 1e3 / warm_iters);
            free(buf);
        }
    }

    printf("\n[warm parallel reads] 64 distinct 4 MiB chunks from page cache, 2 MiB aligned buffers\n");
    {
        size_t chunk = 4u << 20;
        size_t n = 64;
        off_t *offs = malloc(n * sizeof(off_t));
        size_t slots = (file_size - n * chunk) / page;
        off_t base = (off_t)((rng_next() % slots) * page);
        for (size_t i = 0; i < n; i++) offs[i] = base + (off_t)(i * chunk);
        void *buf = aligned_buf(2u << 20, chunk);
        for (size_t i = 0; i < n; i++) pread(fd_warm, buf, chunk, offs[i]);
        free(buf);
        for (size_t ti = 0; ti < 5; ti++) {
            size_t bytes;
            double max_ms;
            double dt = run_parallel(fd_warm, chunk, 2u << 20, threads[ti], offs, n, &bytes, &max_ms);
            printf("threads %2d: %8.2f GB/s\n", threads[ti], bytes / dt / 1e9);
        }
        free(offs);
    }

    munmap(map, file_size);
    close(fd_cold);
    close(fd_warm);
    return 0;
}
