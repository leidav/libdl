#ifndef P_RELU_LAYER_H
#define P_RELU_LAYER_H

#include "layer.h"
#include "update_rule.h"

namespace nn {
class PReLULayer : public Layer {
 public:
  PReLULayer(int layer_size, float alpha = 0.01);
  virtual ~PReLULayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;
  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  Eigen::VectorXf m_alpha;
  AdamUpdate<Eigen::VectorXf> m_alpha_update_rule;
};
};  // namespace nn

#endif
