#include "convolution1x1_layer.h"
#include <utils/convolution_helper/convolution_helper.h>
#include <iostream>

#include <cmath>
#include <random>

namespace nn {
static thread_local std::random_device g_seed_generator;
static thread_local std::mt19937 g_mersenne_twister(g_seed_generator());

Convolution1x1Layer::Convolution1x1Layer(int input_image_width,
                                         int input_image_height,
                                         int input_image_depth,
                                         int output_image_depth, int batch_size,
                                         float regularization_factor)
    : Layer(input_image_width * input_image_height * input_image_depth,
            input_image_width * input_image_height * output_image_depth),
      m_input_width(input_image_width),
      m_input_height(input_image_height),
      m_input_depth(input_image_depth),
      m_batch_size(batch_size),
      m_filter(utils::convolution_helper::filterRows(input_image_depth, 1,
                                                     output_image_depth),
               utils::convolution_helper::filterCols(input_image_depth, 1,
                                                     output_image_depth)),
      m_bias_weights(output_image_depth),
      m_dfilter(m_filter.rows(), m_filter.cols()),
      m_db(m_bias_weights.rows()),
      m_regularization_factor(regularization_factor),
      m_filter_updat_rule(m_filter.rows(), m_filter.cols()),
      m_bias_updat_rule(m_bias_weights.rows()) {
  // Xavier/2 initialization
  std::normal_distribution<float> distribution(
      0.0, sqrtf(2.0f / static_cast<float>(1 * input_image_depth)));
  auto rd = [&distribution]() { return distribution(g_mersenne_twister); };

  m_filter = WeightMatrix::NullaryExpr(m_filter.rows(), m_filter.cols(), rd);
  m_bias_weights = Eigen::VectorXf::NullaryExpr(m_bias_weights.rows(), rd);
}
Convolution1x1Layer::~Convolution1x1Layer() {}

void Convolution1x1Layer::forward(ArrayRef y, const ConstArrayRef &x,
                                  bool train) {
  int output_image_size = y.cols() / m_dfilter.cols();

  const Eigen::Map<const Array> x_reshaped(
      x.data(), m_batch_size * m_input_width * m_input_height, m_input_depth);

  Eigen::Map<Array> y_reshaped(y.data(), m_batch_size * output_image_size,
                               m_filter.cols());

  y_reshaped.matrix().noalias() = (x_reshaped.matrix() * m_filter);

  y_reshaped.matrix().rowwise() += m_bias_weights.transpose();
}

void Convolution1x1Layer::backward(ArrayRef dx, const ConstArrayRef &x,
                                   const ConstArrayRef &y,
                                   const ConstArrayRef &dy) {
  int image_size = y.cols() / m_dfilter.cols();

  Eigen::Map<const Array> dy_reshaped(dy.data(), image_size * m_batch_size,
                                      m_dfilter.cols());
  const Eigen::Map<const Array> x_reshaped(
      x.data(), m_batch_size * m_input_width * m_input_height, m_input_depth);

  Eigen::Map<Array> dx_reshaped(
      dx.data(), m_batch_size * m_input_width * m_input_height, m_input_depth);

  m_dfilter = x_reshaped.matrix().transpose() * dy_reshaped.matrix();

  m_dfilter += m_regularization_factor * m_filter;
  m_db = dy_reshaped.colwise().sum();
  dx_reshaped = dy_reshaped.matrix() * m_filter.transpose();
}

float Convolution1x1Layer::regularizationLoss() {
  return 0.5f * m_regularization_factor *
         (m_filter.array() * m_filter.array()).sum();
}

void Convolution1x1Layer::update(float learning_rate) {
  m_filter_updat_rule.update(m_filter, m_dfilter, learning_rate);
  m_bias_updat_rule.update(m_bias_weights, m_db, learning_rate);
}

uint64_t Convolution1x1Layer::id() {
  return layerNameHash("Convolution1x1Layer");
}

int Convolution1x1Layer::paramCount() { return 8; }

Layer::ArrayRef Convolution1x1Layer::param(int param) {
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
