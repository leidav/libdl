#include "batchnorm_layer.h"
namespace nn {
BatchnormLayer::BatchnormLayer(int batch_size, int layer_size, float alpha)
    : Layer(layer_size, layer_size),
      m_alpha(alpha),
      m_beta(Eigen::RowVectorXf::Zero(layer_size)),
      m_gamma(Eigen::RowVectorXf::Ones((layer_size))),
      m_dbeta(Eigen::RowVectorXf::Zero(layer_size)),
      m_dgamma(Eigen::RowVectorXf::Ones(layer_size)),
      m_batch_mean(layer_size),
      m_batch_var(layer_size),
      m_running_mean(Eigen::RowVectorXf::Zero(layer_size)),
      m_running_var(Eigen::RowVectorXf::Zero(layer_size)),
      m_x_hat(batch_size, layer_size),
      m_beta_update_rule(layer_size),
      m_gamma_update_rule(layer_size) {}

BatchnormLayer::~BatchnormLayer() {}

void BatchnormLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  constexpr float eps = 1e-5f;
  if (train) {
	m_batch_mean = x.colwise().sum() / x.rows();
	m_batch_var = (x.rowwise() - m_batch_mean.array()).colwise().squaredNorm();
	m_running_mean = (1.0f - m_alpha) * m_running_mean + m_alpha * m_batch_mean;
	m_running_var = (1.0f - m_alpha) * m_running_var + m_alpha * m_batch_var;
	m_x_hat = (x.rowwise() - m_batch_mean.array()).rowwise() *
	          (m_batch_var.array() + eps).array().rsqrt();
	y = (m_x_hat.rowwise() * m_gamma.array()).rowwise() + m_beta.array();

  } else {
	y = ((x.rowwise() - m_running_mean.array()).rowwise() *
	     ((m_running_var.array() + eps).array().rsqrt() * m_gamma.array()))
	        .rowwise() +
	    m_beta.array();
  }
}

void BatchnormLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                              const ConstArrayRef &y, const ConstArrayRef &dy) {
  constexpr float eps = 1e-5f;
  // pre allocated temporary array
  static thread_local Layer::Array dx_hat(m_x_hat.rows(), m_x_hat.cols());
  float batch_size = static_cast<float>(x.rows());
  m_dbeta = dy.colwise().sum();
  m_dgamma = (dy * m_x_hat).colwise().sum();
  dx_hat = dy.rowwise() * m_gamma.array();

  // https://kevinzakka.github.io/2016/09/14/batch_normalization/
  // dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0)
  //	- x_hat*np.sum(dxhat*x_hat, axis=0))
  dx = ((dx_hat * batch_size).rowwise() - (dx_hat.colwise().sum())).rowwise() *
           ((1.0f / batch_size) * (m_batch_var.array() + eps).rsqrt()) -
       m_x_hat.rowwise() * (m_x_hat * dx_hat).colwise().sum();
}

void BatchnormLayer::update(float learning_rate) {
  m_beta_update_rule.update(m_beta, m_dbeta, learning_rate);
  // m_beta_update_rule.update(m_beta, m_dbeta, learning_rate);
  m_beta_update_rule.update(m_gamma, m_dgamma, learning_rate);
}

uint64_t BatchnormLayer::id() { return layerNameHash("BatchnormLayer"); }

int BatchnormLayer::paramCount() { return 8; }

Layer::ArrayRef BatchnormLayer::param(int param) {
  switch (param) {
    case 0:
	  return ArrayRef(m_beta);
    case 1:
	  return ArrayRef(m_gamma);
    case 2:
	  return Eigen::Map<Array>(&m_beta_update_rule.t, 1, 1);
    case 3:
	  return ArrayRef(m_beta_update_rule.gradient_average);
    case 4:
	  return ArrayRef(m_beta_update_rule.squared_gradient_average);
    case 5:
	  return Eigen::Map<Array>(&m_gamma_update_rule.t, 1, 1);
    case 6:
	  return ArrayRef(m_gamma_update_rule.gradient_average);
    case 7:
	  return ArrayRef(m_gamma_update_rule.squared_gradient_average);
    default:
	  return Eigen::Map<Array>(nullptr, 0, 0);
  }
}

};  // namespace nn
