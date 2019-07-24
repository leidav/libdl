#ifndef NEURAL_NETWORK_BINDINGS_H
#define NEURAL_NETWORK_BINDINGS_H

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

void createNeuralNetworkBinding(pybind11::module &m);

#endif
