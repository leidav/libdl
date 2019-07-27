#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H

#include "layer.h"

namespace nn {
class MaxPoolingLayer : public Layer {
 public:
  MaxPoolingLayer(int input_image_width, int imput_image_height,
                  int input_image_depth, int output_image_width,
                  int output_image_height, int kernel_size, int batch_size);
  virtual ~MaxPoolingLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  int m_input_width;
  int m_input_height;
  int m_input_depth;
  int m_kernel_size;
  int m_batch_size;
  std::vector<uint8_t> m_indizes;
};
};  // namespace nn

#endif
