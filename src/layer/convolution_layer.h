#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "layer.h"
#include "update_rule.h"

namespace nn {
class ConvolutionLayer : public Layer {
 public:
  using WeightMatrix =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  ConvolutionLayer(int input_image_width, int imput_image_height,
                   int input_image_depth, int output_image_width,
                   int output_image_height, int output_image_depth,
                   int kernel_size, int batch_size, int padding, int stride,
                   float regularization_factor = 1e-6f);
  virtual ~ConvolutionLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  float regularizationLoss() final;

  void update(float learning_rate) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  int m_input_width;
  int m_input_height;
  int m_input_depth;
  int m_kernel_size;
  int m_padding;
  int m_stride;
  int m_batch_size;
  Array m_im2row_x;
  Array m_im2row_dx;
  WeightMatrix m_filter;
  Eigen::VectorXf m_bias_weights;
  WeightMatrix m_dfilter;
  Eigen::VectorXf m_db;
  float m_regularization_factor;
  AdamUpdate<WeightMatrix> m_filter_updat_rule;
  AdamUpdate<Eigen::VectorXf> m_bias_updat_rule;
};
};  // namespace nn

#endif
