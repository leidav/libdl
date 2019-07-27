add_library(dl)
target_sources(dl PUBLIC
	                    "${CMAKE_CURRENT_SOURCE_DIR}/neural_network.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/neural_network.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/param_saver.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/param_saver.cpp"
						#"${CMAKE_CURRENT_SOURCE_DIR}/solver.h"
						#"${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/update_rule.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/convolution_helper/convolution_helper.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/convolution_helper/convolution_helper.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/mnist/mnist_loader.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/utils/mnist/mnist_loader.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/fully_connected_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/fully_connected_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/convolution_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/convolution_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/max_pooling_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/max_pooling_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/average_pooling_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/average_pooling_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/dropout_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/dropout_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/batchnorm_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/batchnorm_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/relu_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/relu_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/tanh_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/tanh_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/sigmoid_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/sigmoid_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/softmax_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/softmax_layer.cpp"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/least_squares_layer.h"
						"${CMAKE_CURRENT_SOURCE_DIR}/layer/least_squares_layer.cpp")

target_compile_features(dl PUBLIC cxx_std_17)
target_link_libraries(dl PUBLIC eigen spdlog OpenMP::OpenMP_CXX)
target_include_directories(dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#optimizations
target_compile_options(dl PUBLIC "-march=native")
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET dl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

