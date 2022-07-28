#include <string>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include "python_nvjpeg_decoder.h"


namespace NVJpegDecoder {

bool PythonDecoder::Init() {
  py::gil_scoped_release release;
  return mDecoder.Init();
}

py::object PythonDecoder::Read(std::string& filename) {
  unsigned char* data = nullptr;
  JpegImage image;
  {
    py::gil_scoped_release release;
    if (mDecoder.Read(filename.c_str(), image)) {
      data = image.Cpu();
    }
  } // gets gil
  
  std::unique_ptr<unsigned char> ret(data);
  if (ret != nullptr) {
    return py::array(py::dtype(py::format_descriptor<uint8_t>::format()), image.Dims(), (void*)ret.get());
  }
  return py::none();
}

py::object PythonDecoder::Decode(std::string& image_bytes) {
  JpegImage image;
  if (!mDecoder.Decode(image_bytes, image)) {
    return py::none();
  }
  std::unique_ptr<unsigned char> data(image.Cpu());
  if (nullptr == data) {
    return py::none();
  }
  return py::array(py::dtype(py::format_descriptor<uint8_t>::format()), image.Dims(), (void*)data.get());
}

PYBIND11_MODULE(pynvjpeg, m) {
  py::class_<PythonDecoder, std::shared_ptr<PythonDecoder>>(m, "Decoder")
     .def(py::init<int>())
     .def("init", &PythonDecoder::Init)
     .def("imread", &PythonDecoder::Read)
     .def("imdecode", &PythonDecoder::Decode);  
}

} // namespace NVJpegDecoder
