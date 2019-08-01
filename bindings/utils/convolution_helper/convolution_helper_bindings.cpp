#include "convolution_helper_bindings.h"
#include <utils/convolution_helper/convolution_helper.h>
#include <tuple>

using namespace nn::utils::convolution_helper;
namespace py = pybind11;

std::tuple<int, int> pyFilterSize(int input_depth, int kernel_size,
                                  int num_kernels) {
  int rows;
  int cols;
  filterSize(rows, cols, input_depth, kernel_size, num_kernels);
  return std::make_tuple<>(rows, cols);
}
std::tuple<int, int> pyIm2rowOutputSize(int image_width, int image_height,
                                        int image_depth, int batch_size,
                                        int kernel_size, int padding,
                                        int stride) {
  int rows;
  int cols;
  im2rowOutputSize(rows, cols, image_width, image_height, image_depth,
                   batch_size, kernel_size, padding, stride);
  return std::make_tuple<>(rows, cols);
}

std::tuple<int, int> pyConvolutionOutputSize(int input_width, int input_height,
                                             int kernel_size, int padding,
                                             int stride) {
  int output_width;
  int output_height;
  convolutionOutputSize(output_width, output_height, input_width, input_height,
                        kernel_size, padding, stride);
  return std::make_tuple<>(output_width, output_height);
}

void createConvolutionHelperBindings(py::module &m) {
  m.def_submodule("convolution_helper")
      .def("im2rowOutputRows", &im2rowOutputRows)
      .def("im2rowOutputCols", &im2rowOutputCols)
      .def("im2rowOutputSize", &pyIm2rowOutputSize)
      .def("im2row", &im2row)
      .def("im2rowBackward", &im2rowBackward)
      .def("filterRows", &filterRows)
      .def("filterCols", &filterCols)
      .def("filterSize", &pyFilterSize)
      .def("padding", &padding)
      .def("convolutionOutputWidth", &convolutionOutputWidth)
      .def("convolutionOutputHeight", &convolutionOutputHeight)
      .def("convolutionOutputSize", &pyConvolutionOutputSize)
      .def("maxPooling", &maxPooling)
      .def("maxPoolingBackward", &maxPoolingBackward)
      .def("averagePooling", &averagePooling)
      .def("averagePoolingBackward", &averagePoolingBackward);
}
