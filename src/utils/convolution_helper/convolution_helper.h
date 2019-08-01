#ifndef IM2ROW_H
#define IM2ROW_H
#include "layer/layer.h"
namespace nn {
namespace utils {
namespace convolution_helper {

///\file
/// \brief helper functions useful for convolution layers and their usage

/// \brief Get the number of rows of the temporary im2row matrix
int im2rowOutputRows(int image_width, int image_height, int image_depth,
                     int batch_size, int kernel_size, int padding, int stride);

/// \brief Get the number of collums of the temporary im2row matrix
int im2rowOutputCols(int image_width, int image_height, int image_depth,
                     int batch_size, int kernel_size, int padding, int stride);

/// \brief Calculate the number of rows and collums of the temporary im2row
/// matrix
void im2rowOutputSize(int &output_rows, int &output_cols, int image_width,
                      int image_height, int image_depth, int batch_size,
                      int kernel_size, int padding, int stride);

/// \brief Creates the im2row temporary matrix used by the convolution layer
/// \param in is an Array of shape
/// (batch_size,image_width*image_height*image_depth)
void im2row(Layer::ArrayRef out, const Layer::ConstArrayRef &in, int width,
            int height, int depth, int batch_size, int kernel_size, int padding,
            int stride);

/// \brief Calculates the im2row backward pass
void im2rowBackward(Layer::ArrayRef dx, const Layer::ConstArrayRef &dy,
                    int width, int height, int depth, int batch_size,
                    int kernel_size, int padding, int stride);

/// \brief Im2row filter matrix rows
int filterRows(int input_depth, int kernel_size, int num_kernels);

/// \brief Im2row filter matrix collumns
int filterCols(int input_depth, int kernel_size, int num_kernels);

/// \brief Im2row filter matrix rows and collumns
void filterSize(int &rows, int &cols, int input_depth, int kernel_size,
                int num_kernels);

/// \brief Required padding if the output size should be the same as the input
/// size
int padding(int kernel_size, bool keep_size);

/// \brief Convolution image output width
int convolutionOutputWidth(int input_width, int input_height, int kernel_size,
                           int padding, int stride);

/// \brief Convolution image output height
int convolutionOutputHeight(int input_width, int input_height, int kernel_size,
                            int padding, int stride);

/// \brief Convolution image output width and height
void convolutionOutputSize(int &output_width, int &output_height,
                           int input_width, int input_height, int kernel_size,
                           int padding, int stride);

/// \brief Max pooling forward
void maxPooling(Layer::ArrayRef out, std::vector<uint8_t> &indices,
                const Layer::ConstArrayRef &in, int width, int height,
                int depth, int batch_size, int kernel_size);

/// \brief Max pooling back propagation
void maxPoolingBackward(Layer::ArrayRef dx, std::vector<uint8_t> &indices,
                        const Layer::ConstArrayRef &dy, int width, int height,
                        int depth, int batch_size, int kernel_size);
/// \brief Max average pooling forward
void averagePooling(Layer::ArrayRef out, const Layer::ConstArrayRef &in,
                    int width, int height, int depth, int batch_size,
                    int kernel_size);

/// \brief Max average pooling back propagation
void averagePoolingBackward(Layer::ArrayRef dx, const Layer::ConstArrayRef &dy,
                            int width, int height, int depth, int batch_size,
                            int kernel_size);

};  // namespace convolution_helper
};  // namespace utils
};  // namespace nn

#endif
