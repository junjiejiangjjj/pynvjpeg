#include "jpeg_image.h"
#include "cuda_util.h"


namespace NVJpegDecoder {

bool JpegImage::Init(int width, int height) {
  unsigned char * pBuffer = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&pBuffer, width * height * NVJPEG_MAX_COMPONENT));

  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
    mNvImage.channel[i] = pBuffer + (width * height * i);
    mNvImage.pitch[i] = (unsigned int)width;
  }
  mNvImage.pitch[0] = (unsigned int)width * 3;
  return true;
}

JpegImage::~JpegImage(){
  cudaFree(mNvImage.channel[0]);
}

} // namespace NVJpegDecoder
