#include "fully_connected_layer.h"
namespace nn {

FullyConnectedLayer::FullyConnectedLayer(int batch_size, int input_size,
                                         int output_size,
                                         float regularization_factor)
    : Layer(batch_size, input_size, output_size),
      m_weights(Eigen::MatrixXf::Identity(input_size, output_size)),
      m_bias_weights(Eigen::VectorXf::Ones(output_size)),
      m_dw(input_size, output_size),
      m_db(output_size),
      m_regularization_factor(regularization_factor) {}
FullyConnectedLayer::~FullyConnectedLayer() {}

void FullyConnectedLayer::forward(const Layer::Array& x) {
  m_y.matrix().noalias() = x.matrix() * m_weights;
  m_y.matrix().rowwise() += m_bias_weights.transpose();
}

void FullyConnectedLayer::backward(const Layer::Array& x,
                                   const Layer::Array& dy) {
  m_dw = x.matrix().transpose() * dy.matrix();
  m_dw += m_regularization_factor * m_weights;
  m_db = dy.rowwise().sum();
  m_dx = dy.matrix() * m_weights.transpose();
}

float FullyConnectedLayer::regularizationLoss() {
  return 0.5f * m_regularization_factor *
         (m_weights.array() * m_weights.array()).sum();
}

};  // namespace nn
