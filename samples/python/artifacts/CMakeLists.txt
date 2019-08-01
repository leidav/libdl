add_custom_target(
	artifacts
	ALL
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/artifacts.py
    ${CMAKE_CURRENT_SOURCE_DIR}/artifacts_single_image.py
    ${CMAKE_CURRENT_SOURCE_DIR}/fetch_and_prepare_data.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/show.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/params.save
	${CMAKE_CURRENT_BINARY_DIR}
	SOURCES artifacts.py
	SOURCES artifacts_single_image.py
	SOURCES fetch_and_prepare_data.sh
	SOURCES show.sh)

add_dependencies(artifacts pydl)

add_custom_command(
  TARGET artifacts POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/samples/python/enviroment.sh
	${CMAKE_CURRENT_BINARY_DIR})

