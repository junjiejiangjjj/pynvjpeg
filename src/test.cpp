#include <string>
#include <sys/timeb.h>
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
    NVJpegDecoder::OriginJpegImages images;
    for (int i = 0; i < 10; i++) {
      images.push_back(image);
    }
    int count = 0;
    timeb t1, t2;
    ftime(&t1);

    while (count < 100) {
      NVJpegDecoder::JpegImages outputs(images.size());
      if (!decoder.BatchDecode(images, outputs)) {
        return -1;
      }
      count++;
    }
    ftime(&t2);
    std::cout << "--------- " << t2.time * 1000 + t2.millitm - t1.time * 1000 - t1.millitm << std::endl;
  }
    
  // test decode    
  {
    timeb t1, t2;
    ftime(&t1);    
    {
      int count = 0;
      while (count < 1000) {
        NVJpegDecoder::JpegImage image;
        if (!decoder.Read(image_file, image)) {
          return -1;
        }
        unsigned char* d = image.Cpu();
        delete[] d;
        count++;
      }
    }
    ftime(&t2);
    std::cout << "--------- " << t2.time * 1000 + t2.millitm - t1.time * 1000 - t1.millitm << std::endl;        
  }
  return 0;
}
