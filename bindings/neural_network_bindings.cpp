#include "neural_network_bindings.h"
#include <neural_network.h>

using namespace nn;
namespace py = pybind11;

void createNeuralNetworkBinding(py::module &m) {
  py::class_<NeuralNetwork> neural_network(m, "NeuralNetwork");
  neural_network.def(py::init<int>())
      .def("addHiddenLayer", &NeuralNetwork::addHiddenLayer)
      .def("addOutputLayer", &NeuralNetwork::addOutputLayer)
      .def("forward", &NeuralNetwork::forward)
      .def("inference", &NeuralNetwork::inference)
      .def("backward", &NeuralNetwork::backward)
      .def("y", &NeuralNetwork::y)
      .def("inference_result", &NeuralNetwork::inference_result);
}
