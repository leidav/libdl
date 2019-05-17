#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>

namespace nn {

class Layer {
 public:
  Layer(int input_size, int output_size);
  virtual ~Layer();
  using Array =
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ArrayRef = Eigen::Ref<Array>;
  using ConstArrayRef = Eigen::Ref<const Array>;

  virtual void forward(ArrayRef y, const ConstArrayRef& x, bool train) = 0;

  int inputSize();
  int outputSize();

  virtual void backward(ArrayRef dx, const ConstArrayRef& x,
                        const ConstArrayRef& y, const ConstArrayRef& dy) = 0;
  virtual float regularizationLoss();

  virtual void update(float learning_rate);

 protected:
  int m_input_size;
  int m_output_size;
};

class OutputLayer : public Layer {
 public:
  OutputLayer(int input_size, int output_size);
  virtual ~OutputLayer();

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  virtual float loss(ArrayRef dx, const ConstArrayRef& x,
                     const ConstArrayRef& labels) = 0;
};
};  // namespace nn

#endif
