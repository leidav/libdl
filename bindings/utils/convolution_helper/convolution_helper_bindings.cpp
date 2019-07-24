#include "convolution_helper_bindings.h"
#include <utils/convolution_helper/convolution_helper.h>

using namespace nn::utils::convolution_helper;
namespace py = pybind11;

void createConvolutionHelperBindings(py::module &m) {
  m.def_submodule("convolution_helper")
      .def("im2rowOutputRows", &im2rowOutputRows)
      .def("im2rowOutputCols", &im2rowOutputCols)
      .def("im2rowOutputSize", &im2rowOutputSize)
      .def("im2row", &im2row)
      .def("im2rowBackward", &im2rowBackward)
      .def("filterRows", &filterRows)
      .def("filterCols", &filterCols)
      .def("filterSize", &filterSize)
      .def("padding", &padding)
      .def("isValid", &isValid)
      .def("imageoutputWidth", &imageoutputWidth)
      .def("imageoutputHeight", &imageoutputHeight)
      .def("imageoutputSize", &imageoutputSize)
      .def("maxPooling", &maxPooling)
      .def("maxPoolingBackward", &maxPoolingBackward)
      .def("averagePooling", &averagePooling)
      .def("averagePoolingBackward", &averagePoolingBackward);
}
