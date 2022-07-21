#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "nvjpeg_decoder.h"


bool ReadImage(const char* filename, std::string& imagedata) {
  std::ifstream input(filename);
  if (!(input.is_open())) {
    std::cout << "Open file " << filename << " failed" << std::endl;
    return false;
  }

  imagedata = std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  return true;
}

int main(int argc, char *argv[]) {
  const char* image_file = argv[1];
  auto decoder = NVJpegDecoder::Decoder();
  if (!decoder.Init()) {
    std::cout << "Init Failed" << std::endl;
    return -1;
  }

  // test batch deocde  
  {
    std::string image;
    if (!ReadImage(image_file, image)) {
      return -1;
    }
    std::cout << image.size() << std::endl;
    NVJpegDecoder::OriginJpegImages images;
    for (int i = 0; i < 20; i++) {
      images.push_back(image);
    }
    NVJpegDecoder::JpegImages outputs(images.size());
    if (!decoder.BatchDecode(images, outputs)) {
      return -1;
    }

    // test decode    
    {
      NVJpegDecoder::JpegImage image;
      if (!decoder.Decode(image_file, image)) {
        return -1;
      }
    }
  }
  return 0;
}
