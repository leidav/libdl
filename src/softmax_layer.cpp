#include "softmax_layer.h"
#include <cmath>
#include <limits>

namespace nn {

static void softmax(const Layer::Array x, Layer::Array& y) {}

SoftmaxLayer::SoftmaxLayer(int batch_size, int layer_size)
    : OutputLayer(batch_size, layer_size, layer_size),
      m_loss(std::numeric_limits<float>::infinity()) {}

SoftmaxLayer::~SoftmaxLayer() {}
void SoftmaxLayer::forward(const Layer::Array& x) {
  // softmax
  // todo: vectorize
  for (int i = 0; i < x.rows(); i++) {
	float max = x.row(i).maxCoeff();
	m_y.row(i) = (x.row(i) - max).exp();
	float sum = m_y.row(i).sum();
	m_y.row(i) /= sum;
  }
}

float SoftmaxLayer::loss(const Eigen::VectorXi& labels) {
  // todo vectorize
  float loss_sum = 0;
  for (int i = 0; i < m_batch_size; i++) {
	int label = labels(i);
	loss_sum += logf(m_y(i, label));
  }
  m_loss = -loss_sum / m_batch_size;
  return m_loss;
}

};  // namespace nn
