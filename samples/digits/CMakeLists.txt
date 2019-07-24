add_executable(digits)
target_sources(digits PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

target_compile_features(digits PUBLIC cxx_std_17)
target_link_libraries(digits PUBLIC dl)
target_include_directories(digits PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#optimizations
target_compile_options(digits PUBLIC "-march=native")
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET digits PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

add_custom_command(
  TARGET digits POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/samples/fetch_data.sh
	$<TARGET_FILE_DIR:digits>)

