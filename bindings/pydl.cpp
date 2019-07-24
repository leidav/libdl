#include <layer/layer_bindings.h>
#include <neural_network_bindings.h>
#include <utils/utils_bindings.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pydl, m)
{
	createLayerBinding(m);
	createNeuralNetworkBinding(m);
	createUtilsBindings(m);
}
