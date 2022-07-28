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

  // test deocde from bytes
  {
    std::string image_btyes;
    if (!ReadImage(image_file, image_btyes)) {
      return -1;
    }
    int count = 0;
    timeb t1, t2;
    ftime(&t1);

    while (count < 1000) {
      NVJpegDecoder::JpegImage image;
      if (!decoder.Decode(image_btyes, image)) {
        return -1;
      }
      count++;
    }
    ftime(&t2);
    std::cout << "--------- " << t2.time * 1000 + t2.millitm - t1.time * 1000 - t1.millitm << std::endl;
  }
    
  // test decode from file
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
