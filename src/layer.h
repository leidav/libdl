#ifndef LAYER_H
#define LAYER_H

#include <Eigen>

namespace nn {
class Layer {
 public:
  Layer(int batch_size, int input_size, int output_size);
  virtual ~Layer();
  using Array =
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  virtual void forward(const Array& x) = 0;
  const Array& y();

  // virtual void backward() = 0;
 protected:
  int m_batch_size;
  int m_input_size;
  int m_output_size;
  Array m_y;
};
};  // namespace nn

#endif
