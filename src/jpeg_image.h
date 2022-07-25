#pragma once
#include <vector>
#include <memory>
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

   bool Init(int width, int height, int channels);

   nvjpegImage_t* GetImagePoint() {
     return mNvImage.get();
   }

   const std::vector<int64_t> Dims() {
     return std::vector<int64_t>{mWidth, mHeight, mChannels};
   }

   unsigned char* Cpu();

private:
   int mWidth = 0;
   int mHeight = 0;
   int mChannels = 0;
   nvjpegChromaSubsampling_t mSubsampling;
   std::unique_ptr<nvjpegImage_t> mNvImage;
};
} // namespace NVJpegDecoder
