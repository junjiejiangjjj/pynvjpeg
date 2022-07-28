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
   Decoder(int device_id=0);
   virtual ~Decoder();
   bool Init();
   bool Read(const char* filename, JpegImage& image);
   bool Decode(std::string& imagedata , JpegImage& image);
   
private:
   bool PrepareJpegImage(const std::string& image, JpegImage& output);
   
private:
   nvjpegHandle_t mHandle;
   nvjpegJpegState_t mState;

   int mDeviceId;
   cudaStream_t mStream;
};

} // namespace NVJpegDecoder
