#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include <vector>

#include "layer/layer.h"

namespace nn {
struct LayerData {
  LayerData() : y(), dx() {}
  LayerData(int batch_size, int input_size, int output_size)
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

  void execute(const Layer::ConstArrayRef& input);

  void backward(const Layer::ConstArrayRef& x, float learning_rate);

  const Layer::ConstArrayRef y();

 private:
  int m_batch_size;
  std::vector<std::unique_ptr<Layer>> m_hidden_layer;
  std::unique_ptr<OutputLayer> m_output_layer;
  std::vector<LayerData> m_hidden_layer_data;
  LayerData m_output_layer_data;
};
};  // namespace nn

#endif
