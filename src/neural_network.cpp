#include "neural_network.h"
#include <spdlog/spdlog.h>

#include <iostream>

namespace nn {

NeuralNetwork::NeuralNetwork(int batch_size) : m_batch_size(batch_size) {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addHiddenLayer(std::shared_ptr<Layer> layer) {
  if ((m_hidden_layer.size() > 0) &&
      (m_hidden_layer.back()->outputSize() != layer->inputSize())) {
	spdlog::error("layer size mismatch");
	std::terminate();
  }
  m_hidden_layer_data.emplace_back(m_batch_size, layer->inputSize(),
                                   layer->outputSize());
  m_hidden_layer_inference_data.emplace_back(1, layer->outputSize());
  m_hidden_layer.push_back(layer);
  //  m_hidden_layer.push_back(std::move(layer));
}

void NeuralNetwork::addOutputLayer(std::shared_ptr<OutputLayer> output_layer) {
  if ((m_hidden_layer.size() > 0) &&
      (m_hidden_layer.back()->outputSize() != output_layer->inputSize())) {
	spdlog::error("layer size mismatch");
	std::terminate();
  }
  m_output_layer_data = LayerTrainData(m_batch_size, output_layer->inputSize(),
                                       output_layer->outputSize());
  m_output_layer_inference_data = Layer::Array(1, output_layer->outputSize());
  m_output_layer = output_layer;
  // m_output_layer = std::move(output_layer);
}

float NeuralNetwork::forward(const Layer::ConstArrayRef &input,
                             const Layer::ConstArrayRef &ground_truth,
                             bool train) {
  float loss = 0;
  Layer::ArrayRef y = m_hidden_layer_data[0].y;
  m_hidden_layer[0]->forward(y, input, train);
  if (train) {
	loss += m_hidden_layer[0]->regularizationLoss();
  }
  for (size_t i = 1; i < m_hidden_layer.size(); i++) {
	const Layer::ConstArrayRef &x = m_hidden_layer_data[i - 1].y;
	Layer::ArrayRef y = m_hidden_layer_data[i].y;
	m_hidden_layer[i]->forward(y, x, train);
	if (train) {
		loss += m_hidden_layer[i]->regularizationLoss();
	}
  }
  m_output_layer->forward(m_output_layer_data.y, m_hidden_layer_data.back().y,
                          train);
  if (train) {
	loss += m_output_layer->regularizationLoss();
  }
  return m_output_layer->loss(m_output_layer_data.dx, m_output_layer_data.y,
                              ground_truth) +
         loss;
}

void NeuralNetwork::inference(const Layer::ConstArrayRef &input) {
  Layer::ArrayRef y = m_hidden_layer_inference_data[0];
  m_hidden_layer[0]->forward(y, input, false);
  for (size_t i = 1; i < m_hidden_layer.size(); i++) {
	const Layer::ConstArrayRef &x = m_hidden_layer_inference_data[i - 1];
	Layer::ArrayRef y = m_hidden_layer_inference_data[i];
	m_hidden_layer[i]->forward(y, x, false);
  }
  m_output_layer->forward(m_output_layer_inference_data,
                          m_hidden_layer_inference_data.back(), false);
}

void NeuralNetwork::backward(const Layer::ConstArrayRef &input,
                             float learning_rate) {
  for (int i = m_hidden_layer.size() - 1; i >= 0; i--) {
	const Layer::ConstArrayRef &dy = (i == m_hidden_layer.size() - 1)
	                                     ? m_output_layer_data.dx
	                                     : m_hidden_layer_data[i + 1].dx;
	const Layer::ConstArrayRef &x =
	    (i == 0) ? input : Layer::ConstArrayRef(m_hidden_layer_data[i - 1].y);
	const Layer::ConstArrayRef &y = m_hidden_layer_data[i].y;
	Layer::Array &dx = m_hidden_layer_data[i].dx;
	m_hidden_layer[i]->backward(dx, x, y, dy);
	m_hidden_layer[i]->update(learning_rate);
  }
}

const Layer::ConstArrayRef NeuralNetwork::y() {
  return Layer::ConstArrayRef(m_output_layer_data.y);
}

const Layer::ConstArrayRef NeuralNetwork::inference_result() {
  return Layer::ConstArrayRef(m_output_layer_inference_data);
}

void NeuralNetwork::openSaveFile(const char *file, int epoch_offset) {
  m_param_writer.open(file);
  m_param_writer.startFile(epoch_offset);
}

void NeuralNetwork::saveParameters(float train_loss, float test_loss) {
  m_param_writer.startEpoch(train_loss, test_loss);
  for (auto layer : m_hidden_layer) {
	saveLayer(*layer, m_param_writer);
  }
  saveLayer(*m_output_layer, m_param_writer);
  m_param_writer.endEpoch();
}

void NeuralNetwork::loadBestParameters(const char *file) {
  ParamReader reader;
  reader.open(file);
  int epoch = reader.bestTestEpoch();
  loadEpoch(reader, epoch);
}

void NeuralNetwork::loadLastParameters(const char *file) {
  ParamReader reader;
  reader.open(file);
  int epoch = reader.epochCount() - 1;
  loadEpoch(reader, epoch);
}

void NeuralNetwork::loadParameters(const char *file, int epoch) {
  ParamReader reader;
  reader.open(file);
  int epochs = reader.epochCount();
  if (epoch > epochs) {
	std::terminate();
  }
  loadEpoch(reader, epoch);
}

void NeuralNetwork::saveLayer(Layer &layer, ParamWriter &writer) {
  uint64_t id = layer.id();
  uint64_t param_count = layer.paramCount();
  writer.startLayer(id, param_count);
  for (int i = 0; i < param_count; i++) {
	writer.addParam(layer.param(i));
  }
}

int NeuralNetwork::loadEpoch(ParamReader &reader, int epoch) {
  reader.setLoadingEpoch(epoch);
  for (auto layer : m_hidden_layer) {
	loadLayer(*layer, reader);
  }
  loadLayer(*m_output_layer, reader);
  return reader.epochOffset();
}

void NeuralNetwork::loadLayer(Layer &layer, ParamReader &reader) {
  uint64_t id;
  int param_count;
  reader.readLayerInfo(id, param_count);
  if ((id != layer.id()) || (param_count != layer.paramCount())) {
	std::terminate();
  }
  for (int i = 0; i < param_count; i++) {
	if (!reader.readParam(layer.param(i))) {
		std::terminate();
	}
  }
}
};  // namespace nn
