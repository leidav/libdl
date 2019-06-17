#ifndef IM2ROW_H
#define IM2ROW_H
#include "layer/layer.h"
namespace nn {
namespace convolution_helper {

int im2rowOutputRows(int image_width, int image_height, int image_depth,
                     int batch_size, int kernel_size, int padding, int stride);

int im2rowOutputCols(int image_width, int image_height, int image_depth,
                     int batch_size, int kernel_size, int padding, int stride);

void im2rowOutputSize(int &output_rows, int &output_cols, int image_width,
                      int image_height, int image_depth, int batch_size,
                      int kernel_size, int padding, int stride);

void im2row(Layer::ArrayRef out, const Layer::ConstArrayRef &in, int width,
            int height, int depth, int batch_size, int kernel_size, int padding,
            int stride);
void im2row_backward(Layer::ArrayRef dx, const Layer::ConstArrayRef &dy,
                     int width, int height, int depth, int batch_size,
                     int kernel_size, int padding, int stride);

int filterRows(int input_depth, int kernel_size, int num_kernels);

int filterCols(int input_depth, int kernel_size, int num_kernels);

void filterSize(int &rows, int &cols, int input_depth, int kernel_size,
                int num_kernels);
int padding(int kernel_size, bool keep_size);

bool isValid(int width, int height, int kernel_size, int padding, int stride);

};  // namespace convolution_helper
};  // namespace nn

#endif
