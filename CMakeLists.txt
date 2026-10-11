cmake_minimum_required(VERSION 3.21)
project(localstack LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)

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
target_include_directories(lstk PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_definitions(lstk PRIVATE CPPHTTPLIB_TCP_NODELAY=1 CPPHTTPLIB_LISTEN_BACKLOG=64)
set_source_files_properties(third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_link_libraries(lstk PRIVATE llama ggml "-framework Foundation" "-framework Metal")
target_compile_options(lstk PRIVATE -Wall -Wextra)
