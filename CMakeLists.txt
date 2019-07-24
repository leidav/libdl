cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
		DESCRIPTION "Deeplearning Library"
        LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		         "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
	    message(STATUS "update submodules")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
			            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
						RESULT_VARIABLE GIT_SUBMOD_RESULT)
		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "could not fetch git submodules")
		endif()
endif()


find_package(OpenMP REQUIRED)

add_subdirectory(extern/eigen)
add_subdirectory(extern/spdlog)
add_subdirectory(extern/pybind11)
add_subdirectory(extern/Catch2)

add_subdirectory(src)
add_subdirectory(bindings)

add_subdirectory(samples/xor)
add_subdirectory(samples/digits)
add_subdirectory(samples/digits_cnn)
add_subdirectory(samples/python/digits)

add_subdirectory(tests)


