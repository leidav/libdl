#ifndef LEAST_SQUARES_LAYER_H
#define LEAST_SQUARES_LAYER_H

#include "layer.h"

// for more documentation see layer.h

namespace nn {
class LeastSquaresLayer : public OutputLayer {
 public:
  /// \param layer_size The number of inputs/outputs per sample
  LeastSquaresLayer(int layer_size);
  virtual ~LeastSquaresLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  float loss(ArrayRef dx, const ConstArrayRef& x,
             const ConstArrayRef& ground_truth) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
