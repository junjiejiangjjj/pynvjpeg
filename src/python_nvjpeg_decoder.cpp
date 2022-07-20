#include <string>
#include <iostream>
#include "python_nvjpeg_decoder.h"


namespace NVJpegDecoder {

bool PythonDecoder::Init() {
  return mDecoder.Init();
}

bool PythonDecoder::BatchDecode(py::list images) {
  for (auto item: images) {
    std::cout << item.attr("__str__")().cast<std::string>() << std::endl;
  }
  return true;
}


PYBIND11_MODULE(pynvjpeg, m) {
  py::class_<PythonDecoder>(m, "Decoder")
     .def(py::init<>())
     .def("init", &PythonDecoder::Init)
     .def("batch_decode", &PythonDecoder::BatchDecode);
}

} // namespace NVJpegDecoder
