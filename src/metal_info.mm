#import <Metal/Metal.h>

#include <cstdint>

extern "C" uint64_t lstk_metal_recommended_working_set(void) {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev ? dev.recommendedMaxWorkingSetSize : 0;
}
