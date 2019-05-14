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

  virtual void forward(const Array& x, bool train) = 0;
  const Array& y();
  const Array& dx();
  int inputSize();
  int outputSize();

  virtual void backward(const Array& x, const Array& dy) = 0;

  virtual float regularizationLoss();

  virtual void update(float learning_rate);

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

  void backward(const Array& x, const Array& dy) final;

  virtual float loss(const Eigen::VectorXf& ground_truth) = 0;
};
};  // namespace nn

#endif
