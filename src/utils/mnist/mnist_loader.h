#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include <layer/layer.h>
namespace nn {
namespace utils {
namespace mnist {

/// \file
/// \brief Small helper functions to load the mnist dataset into Eigen Arrays

/// \brief Load images from a mnist file
/// \param images An EMPTY not allocated Eigen Array.
/// \param path The path to the file
/// \return 0 if loading is successfull, -1 otherwise
/// \note The memory is allocated by the function

int loadImages(nn::Layer::Array& images, const char* path);

/// \brief Load labels from a mnist file
///\param labels An EMPTY not allocated Eigen Array.
/// \param path The path to the file
/// \return 0 if loading is successfull, -1 otherwise
///\note The memory is allocated by the function

int loadLabels(nn::Layer::Array& labels, const char* path);

};  // namespace mnist
};  // namespace utils
};  // namespace nn
#endif
