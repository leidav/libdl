add_executable(dl main.cpp)
target_compile_features(dl PUBLIC cxx_std_17)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET dl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

