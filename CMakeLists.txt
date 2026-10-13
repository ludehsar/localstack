# Build for lstk, the localstack inference engine (Apple Silicon only).
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target lstk -j
#
# llama.cpp (third_party/llama.cpp) is built as a subproject so we get libggml (with the Metal
# backend) and libllama. Our own engine code lives in src/. The standalone programs in tools/
# are not built here; each one's header comment has its own compile line.

cmake_minimum_required(VERSION 3.21)
project(localstack LANGUAGES C CXX OBJC OBJCXX)

# C++20; compile_commands.json lets clangd and other tooling see the exact compile flags.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Configure the llama.cpp subproject: Metal GPU backend with its shader library embedded in the
# binary (no .metallib file to ship), and none of llama.cpp's own tools, tests, examples, or server.
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)


# lstk: the single engine binary (CLI completion, chat, bench, parity checks, and HTTP server).
# httplib.cpp is cpp-httplib's implementation file, vendored inside llama.cpp, used by src/server.cpp.
add_executable(lstk
  src/main.cpp
  src/engine.cpp
  src/expert_store.cpp
  src/gguf_model.cpp
  src/kv_cache.cpp
  src/memory_monitor.cpp
  src/qwen3.cpp
  src/sampler.cpp
  src/server.cpp
  third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp
  src/tokenizer.cpp
  src/metal_info.mm)
# Headers: llama.h (tokenizer + reference model), ggml*.h / gguf.h, and vendor/ for httplib + nlohmann/json.
target_include_directories(lstk PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor)
# TCP_NODELAY so each server-sent-event token chunk is flushed immediately instead of Nagle-batched.
target_compile_definitions(lstk PRIVATE CPPHTTPLIB_TCP_NODELAY=1 CPPHTTPLIB_LISTEN_BACKLOG=64)
# Third-party code: silence its warnings so ours stay readable.
set_source_files_properties(third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp PROPERTIES COMPILE_OPTIONS "-w")
# Foundation + Metal are needed by src/metal_info.mm (Objective-C++).
target_link_libraries(lstk PRIVATE llama ggml "-framework Foundation" "-framework Metal")
target_compile_options(lstk PRIVATE -Wall -Wextra)
