#include "memory_monitor.h"

#include <mach/mach.h>
#include <sys/sysctl.h>

#include <cstdio>

bool memory_snapshot(MemorySnapshot &out) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) != KERN_SUCCESS) return false;
    const uint64_t page = (uint64_t) vm_kernel_page_size;
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
    xsw_usage swap = {};
    len = sizeof(swap);
    sysctlbyname("vm.swapusage", &swap, &len, nullptr, 0);
    out.total = memsize;
    out.free = (uint64_t) vm.free_count * page;
    out.inactive = (uint64_t) vm.inactive_count * page;
    out.wired = (uint64_t) vm.wire_count * page;
    out.compressed = (uint64_t) vm.compressor_page_count * page;
    out.swap_used = swap.xsu_used;
    out.decompressions = vm.decompressions;
    out.swapins = vm.swapins;
    return true;
}

std::string memory_describe(const MemorySnapshot &s) {
    char buf[256];
    snprintf(buf, sizeof(buf), "free %.2f GiB, inactive %.2f GiB, wired %.2f GiB, compressed %.2f GiB, swap used %.2f GiB",
             s.free / 1073741824.0, s.inactive / 1073741824.0, s.wired / 1073741824.0, s.compressed / 1073741824.0,
             s.swap_used / 1073741824.0);
    return buf;
}

std::string memory_pressure_delta(const MemorySnapshot &before, const MemorySnapshot &after, double seconds) {
    char buf[256];
    const double secs = seconds > 0 ? seconds : 1.0;
    snprintf(buf, sizeof(buf), "%.0f decompressions/s, %.0f swap-ins/s, compressed %+.2f GiB, swap %+.2f GiB",
             (double) (after.decompressions - before.decompressions) / secs, (double) (after.swapins - before.swapins) / secs,
             ((double) after.compressed - (double) before.compressed) / 1073741824.0,
             ((double) after.swap_used - (double) before.swap_used) / 1073741824.0);
    return buf;
}
