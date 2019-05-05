#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include <vector>

#include "layer.h"
namespace nn {
class NeuralNetwork {
 public:
  NeuralNetwork();

  ~NeuralNetwork();

  void addLayer(Layer *layer);

  void forward(const Layer::Array &x, Layer::Array &y);

  void backward();

 private:
  std::vector<Layer *> m_layer;
};
};  // namespace nn

#endif
