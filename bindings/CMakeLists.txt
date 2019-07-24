pybind11_add_module(pydl MODULE
	                    "${CMAKE_CURRENT_SOURCE_DIR}/pydl.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/neural_network_bindings.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/neural_network_bindings.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/layer_bindings.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/layer_bindings.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/utils_bindings.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/utils_bindings.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/convolution_helper/convolution_helper_bindings.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/convolution_helper/convolution_helper_bindings.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/mnist/mnist_loader_bindings.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/mnist/mnist_loader_bindings.cpp"
						)
target_link_libraries(pydl PRIVATE dl)
target_include_directories(pydl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	target_compile_options(pydl PRIVATE -fsized-deallocation)
endif()

