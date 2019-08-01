#include "max_pooling_layer.h"
#include <utils/convolution_helper/convolution_helper.h>

#include <cmath>
#include <random>

namespace nn {

MaxPoolingLayer::MaxPoolingLayer(int input_image_width, int input_image_height,
                                 int input_image_depth, int kernel_size,
                                 int batch_size)
    : Layer(input_image_width * input_image_height * input_image_depth,
            utils::convolution_helper::convolutionOutputWidth(
                input_image_width, input_image_height, kernel_size, 0,
                kernel_size) *
                utils::convolution_helper::convolutionOutputHeight(
                    input_image_width, input_image_height, kernel_size, 0,
                    kernel_size) *
                input_image_depth),
      m_input_width(input_image_width),
      m_input_height(input_image_height),
      m_input_depth(input_image_depth),
      m_kernel_size(kernel_size),
      m_batch_size(batch_size),
      m_indizes(static_cast<size_t>(
          m_batch_size *
          utils::convolution_helper::convolutionOutputWidth(
              input_image_width, input_image_height, kernel_size, 0,
              kernel_size) *
          utils::convolution_helper::convolutionOutputHeight(
              input_image_width, input_image_height, kernel_size, 0,
              kernel_size) *
          input_image_depth)) {}
MaxPoolingLayer::~MaxPoolingLayer() {}

void MaxPoolingLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  utils::convolution_helper::maxPooling(y, m_indizes, x, m_input_width,
                                        m_input_height, m_input_depth,
                                        m_batch_size, m_kernel_size);
}

void MaxPoolingLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                               const ConstArrayRef &y,
                               const ConstArrayRef &dy) {
  utils::convolution_helper::maxPoolingBackward(
      dx, m_indizes, dy, m_input_width, m_input_height, m_input_depth,
      m_batch_size, m_kernel_size);
}
uint64_t MaxPoolingLayer::id() { return layerNameHash("MaxPoolingLayer"); }

int MaxPoolingLayer::paramCount() { return 0; }

Layer::ArrayRef MaxPoolingLayer::param(int param) {
  return Eigen::Map<Array>(nullptr, 0, 0);
}

};  // namespace nn
