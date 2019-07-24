#ifndef MNIST_LOADER_BINDINGS_H
#define MNIST_LOADER_BINDINGS_H

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

void createMnistLoaderBindings(pybind11::module &m);
#endif
