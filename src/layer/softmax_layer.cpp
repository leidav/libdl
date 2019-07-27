#include "softmax_layer.h"
#include <cmath>
#include <limits>

namespace nn {

SoftmaxLayer::SoftmaxLayer(int layer_size)
    : OutputLayer(layer_size, layer_size),
      m_loss(std::numeric_limits<float>::infinity()) {}

SoftmaxLayer::~SoftmaxLayer() {}

void SoftmaxLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  // softmax
  for (int i = 0; i < x.rows(); i++) {
	float max = x.row(i).maxCoeff();
	y.row(i) = (x.row(i) - max).exp();
	float sum = y.row(i).sum();
	y.row(i) /= sum;
  }
}

float SoftmaxLayer::loss(ArrayRef dx, const ConstArrayRef &x,
                         const ConstArrayRef &labels) {
  float loss_sum = 0;
  int batch_size = x.rows();
  for (int i = 0; i < batch_size; i++) {
	int label_pos = static_cast<int>(labels(i, 0));
	loss_sum += logf(x(i, label_pos));
	// gradient
	dx.row(i) = x.row(i);
	dx(i, label_pos) -= 1;
  }
  m_loss = -loss_sum / batch_size;
  dx /= batch_size;
  return m_loss;
}

uint64_t SoftmaxLayer::id() { return layerNameHash("SoftmaxLayer"); }
int SoftmaxLayer::paramCount() { return 0; }

Layer::ArrayRef SoftmaxLayer::param(int param) {
  return Eigen::Map<Array>(nullptr, 0, 0);
}

};  // namespace nn
