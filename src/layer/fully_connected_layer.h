#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"
#include "update_rule.h"

namespace nn {
class FullyConnectedLayer : public Layer {
 public:
  using WeightMatrix =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  FullyConnectedLayer(int input_size, int output_size,
                      float regularization_factor = 1e-6f);
  virtual ~FullyConnectedLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  float regularizationLoss() final;

  void update(float learning_rate) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  WeightMatrix m_weights;
  Eigen::VectorXf m_bias_weights;
  WeightMatrix m_dw;
  Eigen::VectorXf m_db;
  float m_regularization_factor;
  AdamUpdate<WeightMatrix> m_weight_updat_rule;
  AdamUpdate<Eigen::VectorXf> m_bias_updat_rule;
};
};  // namespace nn

#endif
