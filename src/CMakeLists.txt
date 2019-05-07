add_library(dl)
target_sources(dl PUBLIC
	                    "${CMAKE_CURRENT_SOURCE_DIR}/neural_network.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/neural_network.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/relu_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/relu_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/sigmoid_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/sigmoid_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/softmax_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/softmax_layer.cpp")

target_compile_features(dl PUBLIC cxx_std_17)
target_link_libraries(dl PUBLIC eigen spdlog pybind11 OpenMP::OpenMP_CXX)
target_include_directories(dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#optimizations
target_compile_options(dl PUBLIC "-march=native")
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET dl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

