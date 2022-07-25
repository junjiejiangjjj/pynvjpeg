#include "jpeg_image.h"
#include "cuda_util.h"


namespace NVJpegDecoder {

bool JpegImage::Init(int width, int height, int channels) {
  mNvImage = std::make_unique<nvjpegImage_t>();
  unsigned char * pBuffer = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&pBuffer, width * height * channels));
  for(int i = 0; i < channels; i++) {
    mNvImage->channel[i] = pBuffer + (width * height * i);
    mNvImage->pitch[i] = (unsigned int)width;
  }
  mNvImage->pitch[0] = (unsigned int)width * channels;
  mWidth = width;
  mHeight = height;
  mChannels = channels;
  return true;
}

JpegImage::JpegImage(JpegImage&& rhs) {
  mWidth = rhs.mWidth;
  mHeight = rhs.mHeight;
  mChannels = rhs.mChannels;  
  mSubsampling = rhs.mSubsampling;
  mNvImage = std::move(rhs.mNvImage);
}

JpegImage& JpegImage::operator=(JpegImage&& rhs) {
  mWidth = rhs.mWidth;
  mHeight = rhs.mHeight;
  mChannels = rhs.mChannels;
  mSubsampling = rhs.mSubsampling;
  mNvImage = std::move(rhs.mNvImage);
  return *this;
}

unsigned char* JpegImage::Cpu() {
  size_t size = mHeight * mWidth * mChannels;
  auto buffer = std::make_unique<unsigned char[]>(size) ;
  CudaStatus s(cudaMemcpy2D(buffer.get(), mWidth * mChannels,
                            mNvImage->channel[0], mNvImage->pitch[0],
                            mWidth * mChannels, mHeight, cudaMemcpyDeviceToHost));

  if (!s.IsOk()) {
    std::cout << "Copy image from GPU to CPU failed: " << s.Msg() << std::endl;
    return nullptr;
  }
  return buffer.release();
}

JpegImage::~JpegImage(){
  if (mNvImage != nullptr) {
    cudaFree(mNvImage->channel[0]);
  }
}


} // namespace NVJpegDecoder
