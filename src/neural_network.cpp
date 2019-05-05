#include "neural_network.h"
namespace nn {

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addLayer(Layer *layer) { m_layer.push_back(layer); }

void NeuralNetwork::forward(const Layer::Array &x, Layer::Array &y) {
  for (auto layer : m_layer) {
	// layer->forward(x, y);
  }
}

void NeuralNetwork::backward() {}

};  // namespace nn
