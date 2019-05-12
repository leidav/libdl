#include "least_squares_layer.h"
#include <cmath>
#include <limits>

namespace nn {

LeastSquaresLayer::LeastSquaresLayer(int batch_size, int layer_size)
    : OutputLayer(batch_size, layer_size, layer_size),
      m_loss(std::numeric_limits<float>::infinity()) {}

LeastSquaresLayer::~LeastSquaresLayer() {}
void LeastSquaresLayer::forward(const Layer::Array& x) { m_y = x; }

float LeastSquaresLayer::loss(const Eigen::VectorXf& ground_truth) {
  float loss_sum = 0;
  for (int i = 0; i < m_batch_size; i++) {
	loss_sum += (m_y.row(i) - ground_truth.array()).matrix().squaredNorm();
  }
  m_loss = loss_sum / m_layer_size * m_batch_size;
  return m_loss;
}

};  // namespace nn
