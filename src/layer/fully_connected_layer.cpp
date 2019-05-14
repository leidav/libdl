#include "fully_connected_layer.h"
#include <chrono>
#include <cmath>
#include <random>

namespace nn {

FullyConnectedLayer::FullyConnectedLayer(int batch_size, int input_size,
                                         int output_size,
                                         float regularization_factor)
    : Layer(batch_size, input_size, output_size),
      m_weights(),
      m_bias_weights(),
      m_dw(input_size, output_size),
      m_db(output_size),
      m_regularization_factor(regularization_factor),
      m_weight_updat_rule(input_size, output_size),
      m_bias_updat_rule(output_size) {
  // Xavier/2 initialization
  // auto t = std::chrono::high_resolution_clock::now();
  // std::mt19937 generator(t.time_since_epoch().count());
  static thread_local std::random_device seed_generator;
  std::mt19937 generator(seed_generator());
  std::normal_distribution<float> distribution(
      0.0, sqrtf(2.0f / static_cast<float>(input_size)));

  auto rd = [&generator, &distribution]() { return distribution(generator); };

  m_weights = WeightMatrix::NullaryExpr(input_size, output_size, rd);
  m_bias_weights = Eigen::VectorXf::NullaryExpr(output_size, rd);
}
FullyConnectedLayer::~FullyConnectedLayer() {}

void FullyConnectedLayer::forward(const Layer::Array& x, bool train) {
  m_y.matrix().noalias() = x.matrix() * m_weights;
  m_y.matrix().rowwise() += m_bias_weights.transpose();
}

void FullyConnectedLayer::backward(const Layer::Array& x,
                                   const Layer::Array& dy) {
  m_dw = x.matrix().transpose() * dy.matrix();
  m_dw += m_regularization_factor * m_weights;
  m_db = dy.colwise().sum();
  m_dx = dy.matrix() * m_weights.transpose();
}

float FullyConnectedLayer::regularizationLoss() {
  return 0.5f * m_regularization_factor *
         (m_weights.array() * m_weights.array()).sum();
}

void FullyConnectedLayer::update(float learning_rate) {
  m_weight_updat_rule.update(m_weights, m_dw, learning_rate);
  m_bias_updat_rule.update(m_bias_weights, m_db, learning_rate);
}

};  // namespace nn
