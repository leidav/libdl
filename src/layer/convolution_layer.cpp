#include "convolution_layer.h"
#include <utils/convolution_helper/convolution_helper.h>
#include <iostream>

#include <cmath>
#include <random>

namespace nn {
static thread_local std::random_device g_seed_generator;
static thread_local std::mt19937 g_mersenne_twister(g_seed_generator());

ConvolutionLayer::ConvolutionLayer(int input_image_width,
                                   int input_image_height,
                                   int input_image_depth,
                                   int output_image_depth, int kernel_size,
                                   int batch_size, int padding, int stride,
                                   float regularization_factor)
    : Layer(input_image_width * input_image_height * input_image_depth,
            utils::convolution_helper::convolutionOutputWidth(
                input_image_width, input_image_height, kernel_size, padding,
                stride) *
                utils::convolution_helper::convolutionOutputHeight(
                    input_image_width, input_image_height, kernel_size, padding,
                    stride) *
                output_image_depth),
      m_input_width(input_image_width),
      m_input_height(input_image_height),
      m_input_depth(input_image_depth),
      m_kernel_size(kernel_size),
      m_padding(padding),
      m_stride(stride),
      m_batch_size(batch_size),
      m_im2row_x(utils::convolution_helper::im2rowOutputRows(
                     input_image_width, input_image_height, input_image_depth,
                     batch_size, kernel_size, padding, stride),
                 utils::convolution_helper::im2rowOutputCols(
                     input_image_width, input_image_height, input_image_depth,
                     batch_size, kernel_size, padding, stride)),
      m_im2row_dx(m_im2row_x.rows(), m_im2row_x.cols()),
      m_filter(utils::convolution_helper::filterRows(
                   input_image_depth, kernel_size, output_image_depth),
               utils::convolution_helper::filterCols(
                   input_image_depth, kernel_size, output_image_depth)),
      m_bias_weights(output_image_depth),
      m_dfilter(m_filter.rows(), m_filter.cols()),
      m_db(m_bias_weights.rows()),
      m_regularization_factor(regularization_factor),
      m_filter_updat_rule(m_filter.rows(), m_filter.cols()),
      m_bias_updat_rule(m_bias_weights.rows()) {
  // Xavier/2 initialization
  std::normal_distribution<float> distribution(
      0.0, sqrtf(2.0f / static_cast<float>(kernel_size * kernel_size *
                                           input_image_depth)));
  auto rd = [&distribution]() { return distribution(g_mersenne_twister); };

  m_filter = WeightMatrix::NullaryExpr(m_filter.rows(), m_filter.cols(), rd);
  m_bias_weights = Eigen::VectorXf::NullaryExpr(m_bias_weights.rows(), rd);
}
ConvolutionLayer::~ConvolutionLayer() {}

void ConvolutionLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  utils::convolution_helper::im2row(m_im2row_x, x, m_input_width,
                                    m_input_height, m_input_depth, m_batch_size,
                                    m_kernel_size, m_padding, m_stride);
  int output_image_size = y.cols() / m_dfilter.cols();
  Eigen::Map<Array> y_reshaped(y.data(), m_batch_size * output_image_size,
                               m_filter.cols());
  y_reshaped.matrix().noalias() = (m_im2row_x.matrix() * m_filter);

  y_reshaped.matrix().rowwise() += m_bias_weights.transpose();
}

void ConvolutionLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                                const ConstArrayRef &y,
                                const ConstArrayRef &dy) {
  int image_size = y.cols() / m_dfilter.cols();

  Eigen::Map<const Array> dy_reshaped(dy.data(), image_size * m_batch_size,
                                      m_dfilter.cols());

  m_dfilter = m_im2row_x.matrix().transpose() * dy_reshaped.matrix();

  m_dfilter += m_regularization_factor * m_filter;
  m_db = dy_reshaped.colwise().sum();
  m_im2row_dx = dy_reshaped.matrix() * m_filter.transpose();
  utils::convolution_helper::im2rowBackward(
      dx, m_im2row_dx, m_input_width, m_input_height, m_input_depth,
      m_batch_size, m_kernel_size, m_padding, m_stride);
}

float ConvolutionLayer::regularizationLoss() {
  return 0.5f * m_regularization_factor *
         (m_filter.array() * m_filter.array()).sum();
}

void ConvolutionLayer::update(float learning_rate) {
  m_filter_updat_rule.update(m_filter, m_dfilter, learning_rate);
  m_bias_updat_rule.update(m_bias_weights, m_db, learning_rate);
}

uint64_t ConvolutionLayer::id() { return layerNameHash("ConvolutionLayer"); }

int ConvolutionLayer::paramCount() { return 8; }

Layer::ArrayRef ConvolutionLayer::param(int param) {
  switch (param) {
    case 0:
	  return ArrayRef(m_filter);
    case 1:
	  return ArrayRef(m_bias_weights);
    case 2:
	  return Eigen::Map<Array>(&m_filter_updat_rule.t, 1, 1);
    case 3:
	  return ArrayRef(m_filter_updat_rule.gradient_average);
    case 4:
	  return ArrayRef(m_filter_updat_rule.squared_gradient_average);
    case 5:
	  return Eigen::Map<Array>(&m_bias_updat_rule.t, 1, 1);
    case 6:
	  return ArrayRef(m_bias_updat_rule.gradient_average);
    case 7:
	  return ArrayRef(m_bias_updat_rule.squared_gradient_average);
    default:
	  return Eigen::Map<Array>(nullptr, 0, 0);
  }
}
};  // namespace nn
