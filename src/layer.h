#ifndef LAYER_H
#define LAYER_H

#include <Eigen>

namespace nn {
class Layer {
 public:
  virtual ~Layer();
  using Array =
	  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  virtual void forward(const Array &x, Array &y) = 0;

  // virtual void backward() = 0;
};
};  // namespace nn

#endif
