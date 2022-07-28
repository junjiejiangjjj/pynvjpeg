#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvjpeg_decoder.h"

namespace py = pybind11;

namespace NVJpegDecoder {

class PythonDecoder {
public:
   PythonDecoder(int device_id=0): mDecoder(device_id) { }

   PythonDecoder(PythonDecoder&) = delete;
   PythonDecoder& operator=(PythonDecoder&) = delete;

   bool Init();
   py::object Read(std::string&);
   py::object Decode(std::string&);

private:
   Decoder mDecoder;
};

} // namespace NVJpegDecoder
