#include "layer_bindings.h"

#include <layer/average_pooling_layer.h>
#include <layer/batchnorm_layer.h>
#include <layer/convolution_layer.h>
#include <layer/dropout_layer.h>
#include <layer/fully_connected_layer.h>
#include <layer/layer.h>
#include <layer/least_squares_layer.h>
#include <layer/max_pooling_layer.h>
#include <layer/relu_layer.h>
#include <layer/sigmoid_layer.h>
#include <layer/softmax_layer.h>
#include <layer/tanh_layer.h>
#include <memory>

using namespace nn;
namespace py = pybind11;

class PyLayer : public Layer {
 public:
  using Layer::Layer;
  void forward(ArrayRef y, const ConstArrayRef& x, bool train) override {
	PYBIND11_OVERLOAD_PURE(void, Layer, forward, y, x, train);
  }

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) override {
	PYBIND11_OVERLOAD_PURE(void, Layer, backward, dx, x, y, dy);
  }
  float regularizationLoss() override {
	PYBIND11_OVERLOAD_PURE(float, Layer, regularizationLoss);
  }

  void update(float learning_rate) override {
	PYBIND11_OVERLOAD_PURE(void, Layer, update, learning_rate);
  }
};

class PyOutputLayer : public OutputLayer {
 public:
  using OutputLayer::OutputLayer;
  void forward(ArrayRef y, const ConstArrayRef& x, bool train) override {
	PYBIND11_OVERLOAD_PURE(void, OutputLayer, forward, y, x, train);
  }

  float regularizationLoss() override {
	PYBIND11_OVERLOAD_PURE(float, OutputLayer, regularizationLoss);
  }

  void update(float learning_rate) override {
	PYBIND11_OVERLOAD_PURE(void, OutputLayer, update, learning_rate);
  }

  float loss(ArrayRef dx, const ConstArrayRef& x,
             const ConstArrayRef& labels) override {
	PYBIND11_OVERLOAD_PURE(float, OutputLayer, loss, dx, x, labels);
  }
};

void createLayerBinding(py::module& m) {
  py::class_<Layer, PyLayer, std::shared_ptr<Layer>> layer(m, "Layer");
  layer.def(py::init<int, int>())
      .def("forward", &Layer::forward)
      .def("backward", &Layer::backward)
      .def("regularizationLoss", &Layer::regularizationLoss)
      .def("update", &Layer::update);

  py::class_<OutputLayer, PyOutputLayer, std::shared_ptr<OutputLayer>>
      output_layer(m, "OutputLayer");
  output_layer.def(py::init<int, int>())
      .def("forward", &OutputLayer::forward)
      .def("backward", &OutputLayer::backward)
      .def("regularizationLoss", &OutputLayer::regularizationLoss)
      .def("update", &OutputLayer::update)
      .def("loss", &OutputLayer::loss);

  py::class_<FullyConnectedLayer, Layer, std::shared_ptr<FullyConnectedLayer>>(
      m, "FullyConnectedLayer")
      .def(py::init<int, int, float>(), py::arg(), py::arg(),
           py::arg("regularization_factor") = 1e-6f)
      .def("forward", &FullyConnectedLayer::forward)
      .def("backward", &FullyConnectedLayer::backward)
      .def("regularizationLoss", &FullyConnectedLayer::regularizationLoss)
      .def("update", &FullyConnectedLayer::update);

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer")
      .def(py::init<int>())
      .def("forward", &ReLULayer::forward)
      .def("backward", &ReLULayer::backward)
      .def("regularizationLoss", &ReLULayer::regularizationLoss)
      .def("update", &ReLULayer::update);

  py::class_<SigmoidLayer, Layer, std::shared_ptr<SigmoidLayer>>(m,
                                                                 "SigmoidLayer")
      .def(py::init<int>())
      .def("forward", &SigmoidLayer::forward)
      .def("backward", &SigmoidLayer::backward)
      .def("regularizationLoss", &SigmoidLayer::regularizationLoss)
      .def("update", &SigmoidLayer::update);

  py::class_<TanhLayer, Layer, std::shared_ptr<TanhLayer>>(m, "TanhLayer")
      .def(py::init<int>())
      .def("forward", &TanhLayer::forward)
      .def("backward", &TanhLayer::backward)
      .def("regularizationLoss", &TanhLayer::regularizationLoss)
      .def("update", &TanhLayer::update);

  py::class_<SoftmaxLayer, OutputLayer, std::shared_ptr<SoftmaxLayer>>(
      m, "SoftmaxLayer")
      .def(py::init<int>())
      .def("forward", &SoftmaxLayer::forward)
      .def("backward", &SoftmaxLayer::backward)
      .def("regularizationLoss", &SoftmaxLayer::regularizationLoss)
      .def("update", &SoftmaxLayer::update)
      .def("loss", &SoftmaxLayer::loss);

  py::class_<LeastSquaresLayer, OutputLayer,
             std::shared_ptr<LeastSquaresLayer>>(m, "LeastSquaresLayer")
      .def(py::init<int>())
      .def("forward", &LeastSquaresLayer::forward)
      .def("backward", &LeastSquaresLayer::backward)
      .def("regularizationLoss", &LeastSquaresLayer::regularizationLoss)
      .def("update", &LeastSquaresLayer::update)
      .def("loss", &LeastSquaresLayer::loss);

  py::class_<BatchnormLayer, Layer, std::shared_ptr<BatchnormLayer>>(
      m, "BatchnormLayer")
      .def(py::init<int, int, float>(), py::arg(), py::arg(),
           py::arg("alpha") = 0.1)
      .def("forward", &BatchnormLayer::forward)
      .def("backward", &BatchnormLayer::backward)
      .def("regularizationLoss", &BatchnormLayer::regularizationLoss)
      .def("update", &BatchnormLayer::update);

  py::class_<DropOutLayer, Layer, std::shared_ptr<DropOutLayer>>(m,
                                                                 "DropOutLayer")
      .def(py::init<int, int>())
      .def("forward", &DropOutLayer::forward)
      .def("backward", &DropOutLayer::backward)
      .def("regularizationLoss", &DropOutLayer::regularizationLoss)
      .def("update", &DropOutLayer::update);

  py::class_<ConvolutionLayer, std::shared_ptr<ConvolutionLayer>>(
      m, "ConvolutionLayer")
      .def(py::init<int, int, int, int, int, int, int, int, int, int, float>(),
           py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
           py::arg(), py::arg(), py::arg(), py::arg(),
           py::arg("regularization_factor") = 1e-6f)
      .def("forward", &ConvolutionLayer::forward)
      .def("backward", &ConvolutionLayer::backward)
      .def("regularizationLoss", &ConvolutionLayer::regularizationLoss)
      .def("update", &ConvolutionLayer::update);

  py::class_<MaxPoolingLayer, Layer, std::shared_ptr<MaxPoolingLayer>>(
      m, "MaxPoolingLayer")
      .def(py::init<int, int, int, int, int, int, int>())
      .def("forward", &MaxPoolingLayer::forward)
      .def("backward", &MaxPoolingLayer::backward)
      .def("regularizationLoss", &MaxPoolingLayer::regularizationLoss)
      .def("update", &MaxPoolingLayer::update);

  py::class_<AveragePoolingLayer, Layer, std::shared_ptr<AveragePoolingLayer>>(
      m, "AveragePoolingLayer")
      .def(py::init<int, int, int, int, int, int, int>())
      .def("forward", &AveragePoolingLayer::forward)
      .def("backward", &AveragePoolingLayer::backward)
      .def("regularizationLoss", &AveragePoolingLayer::regularizationLoss)
      .def("update", &AveragePoolingLayer::update);
}
