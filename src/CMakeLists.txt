add_library(dl dl.cpp)
target_compile_features(dl PUBLIC cxx_std_17)
target_link_libraries(dl PUBLIC eigen spdlog pybind11 OpenMP::OpenMP_CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET dl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

