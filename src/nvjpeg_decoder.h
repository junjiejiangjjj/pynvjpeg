#pragma once

#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include "jpeg_image.h"

namespace NVJpegDecoder {

typedef std::vector<std::string> OriginJpegImages;
typedef std::vector<JpegImage> JpegImages;

class Decoder {
public:
   Decoder();
   virtual ~Decoder();
   bool Init();
   bool BatchDecode(OriginJpegImages& origin_images, JpegImages& outputs, nvjpegOutputFormat_t fmt=NVJPEG_OUTPUT_RGBI);
   
   
private:
   bool PrepareJpegImage(const std::string& image, JpegImage& output);
   
private:
   nvjpegDevAllocator_t mDeviceAllocator;
   nvjpegPinnedAllocator_t mPinnedAllocator;
   nvjpegHandle_t mHandle;
   nvjpegJpegState_t mState;
   nvjpegJpegDecoder_t mNvjpegDecoder;
   nvjpegJpegState_t mNvjpegDecoupledState;
   nvjpegBufferDevice_t mDeviceBuffer;
   nvjpegDecodeParams_t mNvjpegDecodeParams;
   nvjpegBufferPinned_t mPinnedBuffers[2];
   nvjpegJpegStream_t  mJpegStreams[2];
   
   int mBatchSize;
   int mMaxCpuThreads;
   int mDeviceId;
   int mHwDecodeAvailable;
   cudaStream_t mStream;
};

} // namespace NVJpegDecoder