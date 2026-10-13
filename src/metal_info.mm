// metal_info.mm - a one-function Objective-C++ bridge to Metal.
//
// Returns the GPU's recommendedMaxWorkingSetSize: how much memory macOS lets the GPU keep resident
// (13.32 GiB, 74% of RAM, on the 18 GB dev machine). The engine uses it to cap the auto-sized expert pool, and
// main.cpp prints it. Declared extern "C" so plain C++ can call it without Objective-C headers.

#import <Metal/Metal.h>

#include <cstdint>

extern "C" uint64_t lstk_metal_recommended_working_set(void) {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev ? dev.recommendedMaxWorkingSetSize : 0;
}
