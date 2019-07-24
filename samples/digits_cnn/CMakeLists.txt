add_executable(digits_cnn)
target_sources(digits_cnn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

target_compile_features(digits_cnn PUBLIC cxx_std_17)
target_link_libraries(digits_cnn PUBLIC dl)
target_include_directories(digits_cnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#optimizations
target_compile_options(digits_cnn PUBLIC "-march=native")
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET digits PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
add_custom_command(
  TARGET digits_cnn POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/samples/fetch_data.sh
	$<TARGET_FILE_DIR:digits_cnn>)
