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
  int inputSize();
  int outputSize();

  // virtual void backward() = 0;
 protected:
  int m_batch_size;
  int m_input_size;
  int m_output_size;
  Array m_y;
};

class OutputLayer : public Layer {
 public:
  OutputLayer(int batch_size, int input_size, int output_size);
  virtual ~OutputLayer();

  virtual void forward(const Array& x) = 0;

  virtual float loss(const Eigen::VectorXi& labels) = 0;
};
};  // namespace nn

#endif
