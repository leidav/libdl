#include "utils_bindings.h"
#include <utils/convolution_helper/convolution_helper_bindings.h>
#include <utils/mnist/mnist_loader_bindings.h>

namespace py = pybind11;

void createUtilsBindings(py::module &m) {
  py::module utils = m.def_submodule("utils");
  createConvolutionHelperBindings(utils);
  createMnistLoaderBindings(utils);
}
