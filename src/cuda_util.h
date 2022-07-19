#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

namespace NVJpegDecoder {


#define CHECK_CUDA(call)                                                \
  do {                                                                  \
    CudaStatus s(call);                                                 \
    if (!s.IsOk()) {                                                    \
      std::cout << "CUDA Runtime failure: '#" << s.Msg() << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl; \
      return false;                                                     \
    }                                                                   \
  } while (false)                                                       \

#define CHECK_NVJPEG(call)                                              \
  do {                                                                  \
    NvJpegStatus s(call);                                               \
    if (!s.IsOk()) {                                                    \
      std::cout << "NVJPEG failure: '#" << s.Msg() << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl; \
      return false;                                                     \
    }                                                                   \
  } while (false)


int DevMalloc(void **p, size_t s);
int DevFree(void *p);
int HostMalloc(void** p, size_t s, unsigned int f);
int HostFree(void* p);


class CudaStatus {
public:
   explicit CudaStatus(cudaError_t error) : mCode(error) {}
   
   CudaStatus(CudaStatus&) = default;
   CudaStatus& operator=(CudaStatus&) = default;
   
   bool IsOk() {
     return mCode == cudaSuccess;
   }
   
   std::string Msg() {
     return cudaGetErrorString(mCode);
   }

private:
   cudaError_t mCode;
};


class NvJpegStatus {
public:
   explicit NvJpegStatus(nvjpegStatus_t error) : mCode(error) {}

   bool IsOk() {
     return mCode == NVJPEG_STATUS_SUCCESS;
   }

   inline nvjpegStatus_t Code() {return mCode;}

   std::string Msg() {
     switch (mCode) {
     case NVJPEG_STATUS_NOT_INITIALIZED:
       return "NVJPEG_STATUS_NOT_INITIALIZED";
     case NVJPEG_STATUS_INVALID_PARAMETER:
       return "NVJPEG_STATUS_INVALID_PARAMETER";
     case NVJPEG_STATUS_BAD_JPEG:
       return "NVJPEG_STATUS_BAD_JPEG";
     case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
       return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
     case NVJPEG_STATUS_ALLOCATOR_FAILURE:
       return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
     case NVJPEG_STATUS_EXECUTION_FAILED:
       return "NVJPEG_STATUS_EXECUTION_FAILED";
     case NVJPEG_STATUS_ARCH_MISMATCH:
       return "NVJPEG_STATUS_ARCH_MISMATCH";
     case NVJPEG_STATUS_INTERNAL_ERROR:
       return "NVJPEG_STATUS_INTERNAL_ERROR";
     case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
       return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
     case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
       return "NVJPEG_STATUS_INCOMPLETE_BITSTREAM";
     default:
       return "UNKNOWN NVJPEG ERROR";
     }
   }

private:
   nvjpegStatus_t mCode;
};

} // namespace NVJpegDecoder
