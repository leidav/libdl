add_custom_target(
	pydigits
	ALL
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/digits.py
	${CMAKE_CURRENT_BINARY_DIR}
	SOURCES digits.py)

add_dependencies(pydigits pydl)

add_custom_command(
  TARGET pydigits POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/samples/fetch_data.sh
	${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
  TARGET pydigits POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/samples/python/enviroment.sh
	${CMAKE_CURRENT_BINARY_DIR})
