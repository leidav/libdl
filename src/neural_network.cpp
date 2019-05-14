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

float NeuralNetwork::forward(const Layer::Array &input,
                             const Layer::Array &ground_truth, bool train) {
  const Layer::Array *xi = &input;
  float loss = 0;
  for (auto &layer : m_hidden_layer) {
	layer->forward(*xi, train);
	xi = &layer->y();
	loss += layer->regularizationLoss();
  }
  m_output_layer->forward(*xi, train);
  loss += m_output_layer->regularizationLoss();
  return m_output_layer->loss(ground_truth) + loss;
}

void NeuralNetwork::execute(const Layer::Array &input) {
  const Layer::Array *xi = &input;
  for (auto &layer : m_hidden_layer) {
	layer->forward(*xi, false);
	xi = &layer->y();
  }
  m_output_layer->forward(*xi, false);
}

void NeuralNetwork::backward(const Layer::Array &x, float learning_rate) {
  const Layer::Array *dyi = &m_output_layer->dx();
  for (auto i = m_hidden_layer.rbegin(); i != m_hidden_layer.rend() - 1; i++) {
	auto layer = i->get();
	auto next_layer = (i + 1)->get();
	const Layer::Array *dxi = &next_layer->y();
	layer->backward(*dxi, *dyi);
	dyi = &layer->dx();
	layer->update(1e-3f);
  }
  if (m_hidden_layer.size() > 0) {
	m_hidden_layer[0]->backward(x, *dyi);
	m_hidden_layer[0]->update(learning_rate);
  }
}

const Layer::Array &NeuralNetwork::y() { return m_output_layer->y(); }
};  // namespace nn
