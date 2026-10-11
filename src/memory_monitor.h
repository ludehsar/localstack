#pragma once

#include <cstdint>
#include <string>

struct MemorySnapshot {
    uint64_t total = 0;
    uint64_t free = 0;
    uint64_t inactive = 0;
    uint64_t wired = 0;
    uint64_t compressed = 0;
    uint64_t swap_used = 0;
    uint64_t decompressions = 0;
    uint64_t swapins = 0;
    uint64_t available() const { return free + inactive; }
};

bool memory_snapshot(MemorySnapshot &out);
std::string memory_describe(const MemorySnapshot &s);
std::string memory_pressure_delta(const MemorySnapshot &before, const MemorySnapshot &after, double seconds);
