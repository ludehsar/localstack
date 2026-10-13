// memory_monitor.h - snapshots of macOS virtual-memory statistics.
//
// Used to size the expert pool from currently free memory, and to report whether a run pushed
// the machine into compression or swap (which makes benchmark numbers meaningless on 18 GB).

#pragma once

#include <cstdint>
#include <string>

// Byte counts from the Mach VM statistics plus cumulative pressure counters.
struct MemorySnapshot {
    uint64_t total = 0;
    uint64_t free = 0;
    uint64_t inactive = 0;
    uint64_t wired = 0;
    uint64_t compressed = 0;
    uint64_t swap_used = 0;
    // Cumulative event counts since boot; compare two snapshots to get rates.
    uint64_t decompressions = 0;
    uint64_t swapins = 0;
    // Memory we can take without forcing compression: free pages plus inactive (reclaimable) pages.
    uint64_t available() const { return free + inactive; }
};

// Fills `out` from host_statistics64 and sysctl. Returns false if the kernel call fails.
bool memory_snapshot(MemorySnapshot &out);
// One-line summaries: current state, and pressure rates/growth between two snapshots.
std::string memory_describe(const MemorySnapshot &s);
std::string memory_pressure_delta(const MemorySnapshot &before, const MemorySnapshot &after, double seconds);
