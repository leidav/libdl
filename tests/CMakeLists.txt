set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/extern/Catch2/contrib")
include(CTest)
include(Catch)

add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2 dl)

catch_discover_tests(tests)

