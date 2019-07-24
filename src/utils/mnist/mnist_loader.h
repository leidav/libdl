#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include <layer/layer.h>
namespace nn {
namespace utils {
namespace mnist {

int loadImages(nn::Layer::Array& images, const char* path);
int loadLabels(nn::Layer::Array& labels, const char* path);
};  // namespace mnist
};  // namespace utils
};  // namespace nn
#endif
