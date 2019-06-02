#include "convolution_layer.h"
#include <utils/im2row.h>

#include <cmath>
#include <random>

namespace nn {
static thread_local std::random_device g_seed_generator;
static thread_local std::mt19937 g_mersenne_twister(g_seed_generator());

ConvolutionLayer::ConvolutionLayer(
    int input_image_width, int input_image_height, int input_image_depth,
    int output_image_width, int output_image_height, int output_image_depth,
    int kernel_size, int batch_size, int padding, int stride,
    float regularization_factor)
    : Layer(input_image_width * input_image_height * input_image_depth,
            output_image_width * output_image_height * output_image_depth),
      m_input_width(input_image_width),
      m_input_height(input_image_height),
      m_input_depth(input_image_depth),
      m_kernel_size(kernel_size),
      m_padding(kernel_size),
      m_stride(stride),
      m_batch_size(batch_size),
      m_im2row_x(convolution_helper::im2rowOutputRows(
                     input_image_width, input_image_height, input_image_depth,
                     batch_size, kernel_size, padding, stride),
                 convolution_helper::im2rowOutputCols(
                     input_image_width, input_image_height, input_image_depth,
                     batch_size, kernel_size, padding, stride)),
      m_im2row_dx(m_im2row_x.rows(), m_im2row_x.cols()),
      m_filter(convolution_helper::filterRows(input_image_depth, kernel_size,
                                              output_image_depth),
               convolution_helper::filterCols(input_image_depth, kernel_size,
                                              output_image_depth)),
      m_bias_weights(output_image_width * output_image_height *
                     output_image_depth),
      m_dfilter(m_filter.rows(), m_filter.cols()),
      m_db(m_bias_weights.rows()),
      m_regularization_factor(regularization_factor),
      m_filter_updat_rule(m_filter.rows(), m_filter.cols()),
      m_bias_updat_rule(m_bias_weights.rows()) {
  // Xavier/2 initialization
  std::normal_distribution<float> distribution(
      0.0, sqrtf(2.0f / static_cast<float>(input_image_width)));
  auto rd = [&distribution]() { return distribution(g_mersenne_twister); };

  m_filter = WeightMatrix::NullaryExpr(m_filter.rows(), m_filter.cols(), rd);
  m_bias_weights = Eigen::VectorXf::NullaryExpr(m_bias_weights.rows(), rd);
}
ConvolutionLayer::~ConvolutionLayer() {}

void ConvolutionLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  convolution_helper::im2row(m_im2row_x, x, m_input_width, m_input_height,
                             m_input_depth, m_batch_size, m_kernel_size,
                             m_padding, m_stride);
  y.matrix().noalias() = (m_im2row_x.matrix() * m_filter)
                             .reshaped<Eigen::RowMajor>(y.rows(), y.cols());
  y.matrix().rowwise() += m_bias_weights.transpose();
}

void ConvolutionLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                                const ConstArrayRef &y,
                                const ConstArrayRef &dy) {
  int image_size = y.cols() / m_dfilter.cols();
  m_dfilter = m_im2row_x.matrix().transpose() *
              dy.matrix().reshaped<Eigen::RowMajor>(image_size * y.rows(),
                                                    m_dfilter.cols());
  m_dfilter += m_regularization_factor * m_filter;
  m_db = dy.colwise().sum();
  m_im2row_dx.matrix().noalias() =
      dy.matrix().reshaped<Eigen::RowMajor>(image_size * y.rows(),
                                            m_dfilter.cols()) *
      m_filter.transpose();
}

float ConvolutionLayer::regularizationLoss() {
  return 0.5f * m_regularization_factor *
         (m_filter.array() * m_filter.array()).sum();
}

void ConvolutionLayer::update(float learning_rate) {
  m_filter_updat_rule.update(m_filter, m_dfilter, learning_rate);
  m_bias_updat_rule.update(m_bias_weights, m_db, learning_rate);
}

};  // namespace nn
