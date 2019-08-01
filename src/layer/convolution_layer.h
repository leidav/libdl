#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "layer.h"
#include "update_rule.h"

// for more documentation see layer.h

namespace nn {
class ConvolutionLayer : public Layer {
 public:
  using WeightMatrix =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  /// \param input_image_width The with of the input image
  /// \param imput_image_height The height of the input image
  /// \param input_image_depth The number of channels of the input image
  /// \param output_image_depth The number of channels of the output image
  /// \param batch_size The number of input/output images
  /// \param kernel_size The size of the convolution kernel
  /// \param padding The number of pixels the input image is extended per side
  /// \param stride The number of pixels the kernel is moved
  /// \param regularization_factor Small factor for the penalty applied to the
  /// filter kernel to prevent overfitting
  ConvolutionLayer(int input_image_width, int imput_image_height,
                   int input_image_depth, int output_image_depth,
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
