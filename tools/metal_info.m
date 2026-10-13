// metal_info.m - print the Metal GPU limits that bound how much model fits on the GPU.
//
// Most important: recommendedMaxWorkingSetSize, the memory macOS lets the GPU keep resident.
// Weights, KV cache, and expert pool must all fit under it. Results are in docs/baseline.md.
//
//   clang -O2 -fobjc-arc -framework Foundation -framework Metal tools/metal_info.m -o build/tools/metal_info

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <sys/sysctl.h>
#include <unistd.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        uint64_t memsize = 0;
        size_t len = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
        printf("device: %s\n", dev.name.UTF8String);
        printf("physical memory: %llu bytes (%.2f GiB)\n", memsize, memsize / 1073741824.0);
        printf("page size: %d\n", getpagesize());
        printf("hasUnifiedMemory: %d\n", dev.hasUnifiedMemory);
        printf("recommendedMaxWorkingSetSize: %llu bytes (%.2f GiB, %.1f%% of RAM)\n",
               dev.recommendedMaxWorkingSetSize, dev.recommendedMaxWorkingSetSize / 1073741824.0,
               100.0 * dev.recommendedMaxWorkingSetSize / memsize);
        printf("maxBufferLength: %lu bytes (%.2f GiB)\n", (unsigned long)dev.maxBufferLength,
               dev.maxBufferLength / 1073741824.0);
        printf("currentAllocatedSize: %lu bytes\n", (unsigned long)dev.currentAllocatedSize);
        printf("maxThreadgroupMemoryLength: %lu\n", (unsigned long)dev.maxThreadgroupMemoryLength);
        printf("maxThreadsPerThreadgroup: %lu x %lu x %lu\n", (unsigned long)dev.maxThreadsPerThreadgroup.width,
               (unsigned long)dev.maxThreadsPerThreadgroup.height, (unsigned long)dev.maxThreadsPerThreadgroup.depth);
        printf("supportsFamily Apple9 (M3): %d\n", [dev supportsFamily:MTLGPUFamilyApple9]);
        printf("supportsFamily Metal3: %d\n", [dev supportsFamily:MTLGPUFamilyMetal3]);
        printf("supportsBCTextureCompression: %d\n", dev.supportsBCTextureCompression);
    }
    return 0;
}
