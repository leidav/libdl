add_executable(xor)
target_sources(xor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

target_compile_features(xor PUBLIC cxx_std_17)
target_link_libraries(xor PUBLIC dl)
target_include_directories(xor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#optimizations
target_compile_options(xor PUBLIC "-march=native")
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET xor PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

