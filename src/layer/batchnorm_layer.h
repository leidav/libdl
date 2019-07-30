#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "layer.h"
#include "update_rule.h"
namespace nn {
class BatchnormLayer : public Layer {
 public:
  BatchnormLayer(int batch_size, int layer_size, float alpha = 0.1f);
  virtual ~BatchnormLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  void update(float learning_rate) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  float m_alpha;
  Eigen::RowVectorXf m_beta;
  Eigen::RowVectorXf m_gamma;
  Eigen::RowVectorXf m_dbeta;
  Eigen::RowVectorXf m_dgamma;
  Eigen::RowVectorXf m_batch_mean;
  Eigen::RowVectorXf m_batch_var;
  Eigen::RowVectorXf m_running_mean;
  Eigen::RowVectorXf m_running_var;
  Array m_x_hat;
  Array m_dx_hat;
  AdamUpdate<Eigen::RowVectorXf> m_beta_update_rule;
  AdamUpdate<Eigen::RowVectorXf> m_gamma_update_rule;

  // Array m_dx_hat;
};
};  // namespace nn
#endif
