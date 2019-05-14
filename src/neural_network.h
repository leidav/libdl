#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include <vector>

#include "layer/layer.h"

namespace nn {
class NeuralNetwork {
 public:
  NeuralNetwork();

  ~NeuralNetwork();

  void addHiddenLayer(std::unique_ptr<Layer> layer);

  void addOutputLayer(std::unique_ptr<OutputLayer> output_layer);

  float forward(const Layer::Array &input, const Layer::Array &ground_truth,
                bool train);

  void execute(const Layer::Array &input);

  void backward(const Layer::Array &x, float learning_rate);

  const Layer::Array &y();

 private:
  std::vector<std::unique_ptr<Layer>> m_hidden_layer;
  std::unique_ptr<OutputLayer> m_output_layer;
};
};  // namespace nn

#endif
