#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include <vector>

#include "layer/layer.h"

namespace nn {
struct LayerTrainData {
  LayerTrainData() : y(), dx() {}
  LayerTrainData(int batch_size, int input_size, int output_size)
      : y(batch_size, output_size), dx(batch_size, input_size) {}
  Layer::Array y;
  Layer::Array dx;
};

class NeuralNetwork {
 public:
  NeuralNetwork(int batch_size);

  ~NeuralNetwork();

  void addHiddenLayer(std::unique_ptr<Layer> layer);

  void addOutputLayer(std::unique_ptr<OutputLayer> output_layer);

  float forward(const Layer::ConstArrayRef& input,
                const Layer::ConstArrayRef& ground_truth, bool train);

  void inference(const Layer::ConstArrayRef& input);

  void backward(const Layer::ConstArrayRef& x, float learning_rate);

  const Layer::ConstArrayRef y();
  const Layer::ConstArrayRef inference_result();

 private:
  int m_batch_size;
  std::vector<std::unique_ptr<Layer>> m_hidden_layer;
  std::unique_ptr<OutputLayer> m_output_layer;
  std::vector<LayerTrainData> m_hidden_layer_data;
  LayerTrainData m_output_layer_data;
  std::vector<Layer::Array> m_hidden_layer_inference_data;
  Layer::Array m_output_layer_inference_data;
};
};  // namespace nn

#endif
