#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvjpeg_decoder.h"

namespace py = pybind11;

namespace NVJpegDecoder {

class PythonDecoder {
public:
   PythonDecoder() = default;

   PythonDecoder(PythonDecoder&) = delete;
   PythonDecoder& operator=(PythonDecoder&) = delete;

   bool Init();
   bool BatchDecode(py::list images);
   
private:
   Decoder mDecoder;
};

} // namespace NVJpegDecoder
