
#include "cuda_util.h"


namespace NVJpegDecoder {

int DevMalloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int DevFree(void *p) { return (int)cudaFree(p); }
int HostMalloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
int HostFree(void* p) { return (int)cudaFreeHost(p); }

} // namespace NVJpegDecoder
