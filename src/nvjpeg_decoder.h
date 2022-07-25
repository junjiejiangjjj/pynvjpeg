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
   bool Read(const char* filename, JpegImage& image);
   bool Decode(std::string& imagedata , JpegImage& image);   
   bool BatchDecode(OriginJpegImages& origin_images, JpegImages& outputs);
   
private:
   bool PrepareJpegImage(const std::string& image, JpegImage& output);
   bool Destory();
   
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
   cudaStream_t mStream;
};

} // namespace NVJpegDecoder
