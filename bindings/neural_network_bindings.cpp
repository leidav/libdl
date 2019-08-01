#include "neural_network_bindings.h"
#include <neural_network.h>
#include <param_saver.h>

using namespace nn;
namespace py = pybind11;

void createNeuralNetworkBinding(py::module &m) {
  py::class_<NeuralNetwork> neural_network(m, "NeuralNetwork");
  neural_network.def(py::init<int>())
      .def("addHiddenLayer", &NeuralNetwork::addHiddenLayer)
      .def("addOutputLayer", &NeuralNetwork::addOutputLayer)
      .def("addResidual",
           &NeuralNetwork::addResidual)  //(int input);
      .def("forward", &NeuralNetwork::forward)
      .def("inference", &NeuralNetwork::inference)
      .def("backward", &NeuralNetwork::backward)
      .def("result", &NeuralNetwork::result)
      .def("inferenceResult", &NeuralNetwork::inferenceResult)
      .def("openSaveFile", &NeuralNetwork::openSaveFile)
      .def("saveParameters", &NeuralNetwork::saveParameters)
      .def("loadBestParameters", &NeuralNetwork::loadBestParameters)
      .def("loadLastParameters", &NeuralNetwork::loadLastParameters)
      .def("loadParameters", &NeuralNetwork::loadParameters);

  py::class_<ParamWriter> param_writer(m, "ParamWriter");
  param_writer.def(py::init<>())
      .def("open", &ParamWriter::open)
      .def("close", &ParamWriter::close)
      .def("startFile", &ParamWriter::startFile)
      .def("startEpoch", &ParamWriter::startEpoch)
      .def("startLayer", &ParamWriter::startLayer)
      .def("addParam", &ParamWriter::addParam);

  py::class_<ParamReader> param_reader(m, "ParamReader");
  param_reader.def(py::init<>())
      .def("open", &ParamReader::open)
      .def("close", &ParamReader::close)
      .def("epochCount", &ParamReader::epochCount)
      .def("setLoadingEpoch", &ParamReader::setLoadingEpoch)
      .def("readLayerInfo", &ParamReader::readLayerInfo)
      .def("readParam", &ParamReader::readParam)
      .def("epochLosses", &ParamReader::epochLosses)
      .def("bestTrainEpoch", &ParamReader::bestTrainEpoch)
      .def("bestTestEpoch", &ParamReader::bestTestEpoch)
      .def("epochOffset", &ParamReader::epochOffset);
}
