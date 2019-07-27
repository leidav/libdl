#include "least_squares_layer.h"
#include <cmath>
#include <limits>

namespace nn {

LeastSquaresLayer::LeastSquaresLayer(int layer_size)
    : OutputLayer(layer_size, layer_size),
      m_layer_size(layer_size),
      m_loss(std::numeric_limits<float>::infinity()) {}

LeastSquaresLayer::~LeastSquaresLayer() {}

void LeastSquaresLayer::forward(ArrayRef y, const ConstArrayRef &x,
                                bool train) {
  y = x;
}

float LeastSquaresLayer::loss(ArrayRef dx, const ConstArrayRef &x,
                              const ConstArrayRef &ground_truth) {
  float loss_sum = 0;
  int batch_size = x.rows();
  for (int i = 0; i < batch_size; i++) {
	loss_sum += (x.row(i) - ground_truth.row(i)).matrix().squaredNorm();
  }
  m_loss = loss_sum / (m_layer_size * batch_size);
  dx = 2 * (x - ground_truth);
  return m_loss;
}
uint64_t LeastSquaresLayer::id() { return layerNameHash("LeastSquaresLayer"); }

int LeastSquaresLayer::paramCount() { return 0; }

Layer::ArrayRef LeastSquaresLayer::param(int param) {
  return Eigen::Map<Array>(nullptr, 0, 0);
}

};  // namespace nn
