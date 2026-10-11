# localstack

An LLM inference engine for Apple Silicon that is being built to run Mixture-of-Experts models
larger than RAM by keeping the always-used weights resident and streaming expert weights from
the SSD through a managed cache. C++20 on top of ggml's Metal kernels (vendored through the
llama.cpp submodule). See `docs/plan.md` for the research, design, and phase plan, and
`docs/baseline.md` for measurements on the development machine (MacBook Pro M3 Pro, 18 GB).

## Build

```sh
git submodule update --init --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lstk -j
```

## Run

```sh
./build/lstk --model ~/Models/gguf/Qwen3-1.7B-Q8_0.gguf --chat
./build/lstk --model model.gguf --prompt "Once upon a time" --n-predict 64 --temp 0.7
./build/lstk --model model.gguf --bench 256 64
./build/lstk --model model.gguf --dump-ref ref.bin --n-predict 32
./build/lstk --model model.gguf --compare-ref ref.bin
./build/lstk --model moe.gguf --chat --trace-routing trace.bin && python3 tools/cache_sim.py trace.bin
./build/lstk --model moe.gguf --stream-experts --pool-gib 4 --chat
./build/lstk --model model.gguf --serve --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
./tools/test_streaming.sh
./tools/moe_check.sh ~/Models/gguf/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
```

Supported architectures so far: qwen2, qwen3 (dense), qwen3moe (resident or streamed experts).

## Tools

- `tools/metal_info.m` prints the Metal working-set cap and buffer limits.
- `tools/ssd_bench.c` measures cold and warm SSD read throughput by chunk size and thread count.
- `tools/gguf_info.c` dumps GGUF metadata and tensor layout.
- `tools/cache_sim.py` replays a routing trace against expert-cache policies.
- `tools/fetch_models.sh` downloads the GGUF files used by the plan with resume support.
- `tools/make_tiny_moe.py` writes a synthetic Qwen3-MoE GGUF; `tools/test_streaming.sh` checks streaming against resident logits with it.
- `tools/moe_check.sh` runs the streaming self-consistency, trace, simulator, and speed sweep on a real MoE GGUF into `docs/runs/`.
