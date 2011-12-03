include_directories( include )
include_directories( "../include" )

file( GLOB_RECURSE TEST_HEADERS "include/*.h" )
file( GLOB_RECURSE TEST_SOURCES "src/*.cpp" )

# OpenMM
set( OPENMM_DIR "" CACHE PATH "OpenMM Install Directory")
if( OPENMM_DIR )
	link_directories( "${OPENMM_DIR}/lib/" )
	include_directories( "${OPENMM_DIR}/include/" )
endif()

find_library( OPENMM_SERIALIAZE_LIB "OpenMMSerialization" HINT ${OPENMM_DIR}/lib )

# Build
add_executable( LTMDTest main.cpp ${TEST_SOURCES} ${TEST_HEADERS} )
target_link_libraries( LTMDTest "OpenMMLTMD" "LTMDReference" "cppunit" ${OPENMM_SERIALIAZE_LIB} )

# Copy files
if( CMAKE_GENERATOR MATCHES "Xcode" )
	add_custom_command(
		TARGET LTMDTest
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/ ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/data/
	)
else()
	add_custom_command(
		TARGET LTMDTest
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/ ${CMAKE_CURRENT_BINARY_DIR}/data/
	)
endif()
