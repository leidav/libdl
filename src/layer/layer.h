#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>

namespace nn {
class Layer {
 public:
  Layer(int batch_size, int input_size, int output_size);
  virtual ~Layer();
  using Array =
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  virtual void forward(const Array& x) = 0;
  const Array& y();
  const Array& dx();
  int inputSize();
  int outputSize();

  virtual void backward(const Array& x, const Array& dy) = 0;

  virtual float regularizationLoss();

 protected:
  int m_batch_size;
  int m_input_size;
  int m_output_size;
  Array m_y;
  Array m_dx;
};

class OutputLayer : public Layer {
 public:
  OutputLayer(int batch_size, int input_size, int output_size);
  virtual ~OutputLayer();

  virtual void forward(const Array& x) = 0;

  void backward(const Array& x, const Array& dy) final;

  virtual float loss(const std::vector<int>& labels) = 0;
};
};  // namespace nn

#endif
