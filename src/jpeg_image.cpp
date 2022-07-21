#include "jpeg_image.h"
#include "cuda_util.h"


namespace NVJpegDecoder {

bool JpegImage::Init(int width, int height) {
  mNvImage = new nvjpegImage_t;
  unsigned char * pBuffer = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&pBuffer, width * height * NVJPEG_MAX_COMPONENT));
  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
    mNvImage->channel[i] = pBuffer + (width * height * i);
    mNvImage->pitch[i] = (unsigned int)width;
  }
  mNvImage->pitch[0] = (unsigned int)width * 3;
  return true;
}

JpegImage::JpegImage(JpegImage&& rhs) {
  mWidth = rhs.mWidth;
  mHeight = rhs.mHeight;
  mSubsampling = rhs.mSubsampling;
  mNvImage = rhs.mNvImage;
  rhs.mNvImage = nullptr;
}

JpegImage& JpegImage::operator=(JpegImage&& rhs) {
  mWidth = rhs.mWidth;
  mHeight = rhs.mHeight;
  mSubsampling = rhs.mSubsampling;
  mNvImage = rhs.mNvImage;
  rhs.mNvImage = nullptr;
  return *this;
}

JpegImage::~JpegImage(){
  if (mNvImage != nullptr) {
    cudaFree(mNvImage->channel[0]);
    delete mNvImage;
    mNvImage = nullptr;
  }
}

} // namespace NVJpegDecoder
