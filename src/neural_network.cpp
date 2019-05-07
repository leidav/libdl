#include "neural_network.h"
#include <spdlog/spdlog.h>
namespace nn {

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addHiddenLayer(std::unique_ptr<Layer> layer) {
  if ((m_hidden_layer.size() > 0) &&
      (m_hidden_layer.back()->outputSize() != layer->inputSize())) {
	spdlog::error("layer size mismatch");
	std::terminate();
  }
  m_hidden_layer.push_back(std::move(layer));
}

void NeuralNetwork::addOutputLayer(std::unique_ptr<OutputLayer> output_layer) {
  if ((m_hidden_layer.size() > 0) &&
      (m_hidden_layer.back()->outputSize() != output_layer->inputSize())) {
	spdlog::error("layer size mismatch");
	std::terminate();
  }
  m_output_layer = std::move(output_layer);
}

void NeuralNetwork::forward(const Layer::Array &x) {
  const Layer::Array *xi = &x;
  for (auto &layer : m_hidden_layer) {
	layer->forward(*xi);
	xi = &layer->y();
  }
  m_output_layer->forward(*xi);
}

void NeuralNetwork::backward() {}

float NeuralNetwork::loss(const Eigen::VectorXi &labels) {
  return m_output_layer->loss(labels);
}

const Layer::Array &NeuralNetwork::y() { return m_output_layer->y(); }
};  // namespace nn
