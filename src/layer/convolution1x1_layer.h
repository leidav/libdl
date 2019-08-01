#ifndef CONVOLUTION1x1_LAYER_H
#define CONVOLUTION1x1_LAYER_H

#include "layer.h"
#include "update_rule.h"

// for more documentation see layer.h

namespace nn {

///\brief Fast convolution Layer with kernel size = 1, stride = 1 and padding =
/// 0
class Convolution1x1Layer : public Layer {
 public:
  using WeightMatrix =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  /// \param input_image_width The with of the input image
  /// \param imput_image_height The height of the input image
  /// \param input_image_depth The number of channels of the input image
  /// \param output_image_width The with of the output image
  /// \param output_image_height The height of the output image
  /// \param output_image_depth The number of channels of the output image
  /// \param batch_size The number of input/output images
  /// \param regularization_factor Small factor for the penalty applied to the
  /// filter kernel to prevent overfitting
  Convolution1x1Layer(int input_image_width, int imput_image_height,
                      int input_image_depth, int output_image_depth,
                      int batch_size, float regularization_factor = 1e-6f);
  virtual ~Convolution1x1Layer();

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
  int m_batch_size;
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
