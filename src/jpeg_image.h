#pragma once

#include <nvjpeg.h>

namespace NVJpegDecoder {

class JpegImage {

public:
   JpegImage() = default;
   ~JpegImage();

   JpegImage(const JpegImage&) = delete;
   JpegImage& operator=(const JpegImage&) = delete;

   bool Init(int width, int height);

   nvjpegImage_t* GetImagePoint() {
     return &mNvImage;
   }

private:
   int mWidth;
   int mHeight;
   nvjpegChromaSubsampling_t mSubsampling;
   nvjpegImage_t mNvImage;
};
} // namespace NVJpegDecoder
