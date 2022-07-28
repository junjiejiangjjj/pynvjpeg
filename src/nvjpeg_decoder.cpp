#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include "cuda_util.h"
#include "nvjpeg_decoder.h"


namespace NVJpegDecoder {

Decoder::Decoder(int device_id): mDeviceId(device_id){
} 

bool Decoder::Init() {
  if (mDeviceId < 0) {
    std::cout<< "Device id must >= 0, the input is "
             << mDeviceId << std::endl;
    return false;
  }

  CHECK_CUDA(cudaSetDevice(mDeviceId));
  CHECK_NVJPEG(nvjpegCreateSimple(&mHandle));
  CHECK_NVJPEG(nvjpegJpegStateCreate(mHandle, &mState));
  CHECK_CUDA(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
  return true;
}

bool Decoder::Read(const char* filename, JpegImage& image) {
  std::ifstream input(filename);
  if (!(input.is_open())) {
    std::cout << "Open file " << filename << " failed" << std::endl;
    return false;
  }
  std::string imagedata((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (!Decode(imagedata, image)) {
    return false;
  }
  return true;
}

bool Decoder::Decode(std::string& imagedata , JpegImage& image) {
  if (!PrepareJpegImage(imagedata, image)) {
    return false;
  }
  CHECK_NVJPEG(nvjpegDecode(
    mHandle,
    mState,
    (const unsigned char *)imagedata.data(),
    imagedata.size(),
    NVJPEG_OUTPUT_RGBI,
    image.GetImagePoint(),
    mStream));
  return true;
}

bool Decoder::PrepareJpegImage(const std::string& image, JpegImage& output) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  CHECK_NVJPEG(nvjpegGetImageInfo(
                 mHandle, (unsigned char *)image.data(), image.size(),
                 &channels, &subsampling, widths, heights));
  
  if (NVJPEG_CSS_UNKNOWN == subsampling) {
    std::cout << "Unknown chroma subsampling" << std::endl;
    return false;
  }

  if (!output.Init(widths[0], heights[0], channels)) {
    return false;
  }
  return true;
}

Decoder::~Decoder() {
  cudaStreamDestroy(mStream);
  nvjpegJpegStateDestroy(mState);  
}

} // namespace NVJpegDecoder
