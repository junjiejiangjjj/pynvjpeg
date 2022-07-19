#include <string>
#include <iostream>
#include <cassert>
#include "cuda_util.h"
#include "nvjpeg_decoder.h"


namespace NVJpegDecoder {

Decoder::Decoder():mDeviceAllocator{&DevMalloc, &DevFree},
                   mPinnedAllocator{&HostMalloc, &HostFree},
                   mHwDecodeAvailable(true){
} 

bool Decoder::Init() {
  auto status = NvJpegStatus(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &mDeviceAllocator,
                                            &mPinnedAllocator,NVJPEG_FLAGS_DEFAULT, &mHandle));
  if (!status.IsOk()) {
    if (status.Code() == NVJPEG_STATUS_ARCH_MISMATCH) {
      std::cout<<"Hardware Decoder not supported. Falling back to default backend"<<std::endl;
      CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &mDeviceAllocator,
                                  &mPinnedAllocator,NVJPEG_FLAGS_DEFAULT, &mHandle));
    }
  } else {
    mHwDecodeAvailable = true;
  }

  CHECK_NVJPEG(nvjpegJpegStateCreate(mHandle, &mState));
  CHECK_NVJPEG(nvjpegDecoderCreate(mHandle, NVJPEG_BACKEND_DEFAULT, &mNvjpegDecoder));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(mHandle, mNvjpegDecoder, &mNvjpegDecoupledState));

  // create_decoupled_api_handles
  CHECK_NVJPEG(nvjpegDecoderCreate(mHandle, NVJPEG_BACKEND_DEFAULT, &mNvjpegDecoder));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(mHandle, mNvjpegDecoder, &mNvjpegDecoupledState));
  CHECK_NVJPEG(nvjpegBufferPinnedCreate(mHandle, NULL, &mPinnedBuffers[0]));
  CHECK_NVJPEG(nvjpegBufferPinnedCreate(mHandle, NULL, &mPinnedBuffers[1]));
  CHECK_NVJPEG(nvjpegBufferDeviceCreate(mHandle, NULL, &mDeviceBuffer));
  CHECK_NVJPEG(nvjpegJpegStreamCreate(mHandle, &mJpegStreams[0]));
  CHECK_NVJPEG(nvjpegJpegStreamCreate(mHandle, &mJpegStreams[1]));
  CHECK_NVJPEG(nvjpegDecodeParamsCreate(mHandle, &mNvjpegDecodeParams));
  CHECK_CUDA(
    cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
  return true;
}

bool Decoder::BatchDecode(OriginJpegImages& images, JpegImages& outputs, nvjpegOutputFormat_t fmt) {
  if (NVJPEG_OUTPUT_RGBI != fmt) {
    std::cout << "Only support NVJPEG_OUTPUT_RGBI" << std::endl;
    return false;
  }
  
  CHECK_CUDA(cudaStreamSynchronize(mStream));
  std::vector<const unsigned char*> bitstreams;
  std::vector<size_t> bitstreams_size;
  
  for (size_t i = 0; i < images.size(); i++) {
    bitstreams.push_back((const unsigned char *)images[i].data());
    bitstreams_size.push_back(images[i].size());
    std::cout << images[i].size() << std::endl;
  }
  
  CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(mNvjpegDecoupledState, mDeviceBuffer));
  int buffer_index = 0;
  CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(mNvjpegDecodeParams, fmt));
  
  for (size_t i = 0; i < images.size(); i++) {
    
    CHECK_NVJPEG(
      nvjpegJpegStreamParse(mHandle, bitstreams[i], bitstreams_size[i],
                            0, 0, mJpegStreams[buffer_index]));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(mNvjpegDecoupledState,
                                               mPinnedBuffers[buffer_index]));
    CHECK_NVJPEG(nvjpegDecodeJpegHost(mHandle, mNvjpegDecoder, mNvjpegDecoupledState,
                                      mNvjpegDecodeParams, mJpegStreams[buffer_index]));
    CHECK_CUDA(cudaStreamSynchronize(mStream));
    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(mHandle, mNvjpegDecoder, mNvjpegDecoupledState,
                                                  mJpegStreams[buffer_index], mStream));
    buffer_index = 1 - buffer_index;
    if (!PrepareJpegImage(images[i], outputs[i])) {
      return false;
    }
    CHECK_NVJPEG(nvjpegDecodeJpegDevice(mHandle, mNvjpegDecoder, mNvjpegDecoupledState,
                                        outputs[i].GetImagePoint(), mStream));
  }
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

  // for NVJPEG_OUTPUT_RGBI, the channels is always 1;
  if (!output.Init(widths[0], heights[0])) {
    return false;
  }
  return true;
}

Decoder::~Decoder() {
  cudaStreamDestroy(mStream);
}

} // namespace NVJpegDecoder
