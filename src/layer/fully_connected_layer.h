#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"
#include "update_rule.h"

// for more documentation see layer.h

namespace nn {
class FullyConnectedLayer : public Layer {
 public:
  using WeightMatrix =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  /// \param input_size The output size of the previous layer
  /// \param output_size The input size of the next layer
  /// \param regularization_factor Small factor for the penalty applied to the
  /// filter kernel to prevent overfitting
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
