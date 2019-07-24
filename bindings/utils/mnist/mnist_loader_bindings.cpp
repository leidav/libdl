#include "mnist_loader_bindings.h"
#include <utils/mnist/mnist_loader.h>

using namespace nn::utils::mnist;
namespace py = pybind11;

static nn::Layer::Array pyLoadImages(const char* path) {
  nn::Layer::Array images;
  loadImages(images, path);
  return images;
}
static nn::Layer::Array pyLoadLabels(const char* path) {
  nn::Layer::Array labels;
  loadLabels(labels, path);
  nn::Layer::Array();
  return labels;
}

void createMnistLoaderBindings(py::module& m) {
  m.def_submodule("mnist")
      .def("loadImages", &pyLoadImages)
      .def("loadLabels", &pyLoadLabels);
}
