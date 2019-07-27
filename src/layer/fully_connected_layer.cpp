#include "fully_connected_layer.h"
#include <cmath>
#include <random>

namespace nn {
static thread_local std::random_device g_seed_generator;
static thread_local std::mt19937 g_mersenne_twister(g_seed_generator());

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size,
                                         float regularization_factor)
    : Layer(input_size, output_size),
      m_weights(),
      m_bias_weights(),
      m_dw(input_size, output_size),
      m_db(output_size),
      m_regularization_factor(regularization_factor),
      m_weight_updat_rule(input_size, output_size),
      m_bias_updat_rule(output_size) {
  // Xavier/2 initialization
  std::normal_distribution<float> distribution(
      0.0, sqrtf(2.0f / static_cast<float>(input_size)));
  auto rd = [&distribution]() { return distribution(g_mersenne_twister); };

  m_weights = WeightMatrix::NullaryExpr(input_size, output_size, rd);
  m_bias_weights = Eigen::VectorXf::NullaryExpr(output_size, rd);
}
FullyConnectedLayer::~FullyConnectedLayer() {}

void FullyConnectedLayer::forward(ArrayRef y, const ConstArrayRef &x,
                                  bool train) {
  y.matrix().noalias() = x.matrix() * m_weights;
  y.matrix().rowwise() += m_bias_weights.transpose();
}

void FullyConnectedLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                                   const ConstArrayRef &y,
                                   const ConstArrayRef &dy) {
  m_dw = x.matrix().transpose() * dy.matrix();
  m_dw += m_regularization_factor * m_weights;
  m_db = dy.colwise().sum();
  dx = dy.matrix() * m_weights.transpose();
}

float FullyConnectedLayer::regularizationLoss() {
  return 0.5f * m_regularization_factor *
         (m_weights.array() * m_weights.array()).sum();
}

void FullyConnectedLayer::update(float learning_rate) {
  m_weight_updat_rule.update(m_weights, m_dw, learning_rate);
  m_bias_updat_rule.update(m_bias_weights, m_db, learning_rate);
}

uint64_t FullyConnectedLayer::id() {
  return layerNameHash("FullyConnectedLayer");
}

int FullyConnectedLayer::paramCount() { return 8; }

Layer::ArrayRef FullyConnectedLayer::param(int param) {
  switch (param) {
    case 0:
	  return ArrayRef(m_weights);
    case 1:
	  return ArrayRef(m_bias_weights);
    case 2:
	  return Eigen::Map<Array>(&m_weight_updat_rule.t, 1, 1);
    case 3:
	  return ArrayRef(m_weight_updat_rule.gradient_average);
    case 4:
	  return ArrayRef(m_weight_updat_rule.squared_gradient_average);
    case 5:
	  return Eigen::Map<Array>(&m_bias_updat_rule.t, 1, 1);
    case 6:
	  return ArrayRef(m_bias_updat_rule.gradient_average);
    case 7:
	  return ArrayRef(m_bias_updat_rule.squared_gradient_average);
    default:
	  return Eigen::Map<Array>(nullptr, 0, 0);
  }
}

};  // namespace nn
