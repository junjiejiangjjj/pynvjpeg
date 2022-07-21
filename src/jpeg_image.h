#pragma once

#include <nvjpeg.h>

namespace NVJpegDecoder {

class JpegImage {

public:
   JpegImage() noexcept : mNvImage(nullptr) {}
   virtual ~JpegImage();

   JpegImage(const JpegImage&) = delete;
   JpegImage& operator=(const JpegImage&) = delete;

   JpegImage(JpegImage&& rhs);
   JpegImage& operator=(JpegImage&& rhs);

   bool Init(int width, int height);

   nvjpegImage_t* GetImagePoint() {
     return mNvImage;
   }

private:
   int mWidth;
   int mHeight;
   nvjpegChromaSubsampling_t mSubsampling;
   nvjpegImage_t* mNvImage;
};
} // namespace NVJpegDecoder
