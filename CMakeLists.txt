cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
		DESCRIPTION "Deeplearning Library"
        LANGUAGES CXX)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		         "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(src)

