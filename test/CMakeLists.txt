include_directories( include )
include_directories( "../include" )

set( TEST_HEADERS "include/ReferenceTest.h" )
set( TEST_SOURCES "src/ReferenceTest.cpp" )
set( LIBS "LTMDReference" )

option( BUILD_GPU "Build GPU code" Off )
if( BUILD_GPU )
	option( BUILD_GPU_CUDA "Build Cuda GPU code" Off )
	if( BUILD_GPU_CUDA )
		list( APPEND TEST_HEADERS "include/CudaTest.h" )
		list( APPEND TEST_SOURCES "src/CudaTest.cpp" )
		list( APPEND LIBS "LTMDCuda" )
	endif( BUILD_GPU_CUDA )
endif( BUILD_GPU ) 

# OpenMM
set( OPENMM_DIR "" CACHE PATH "OpenMM Install Directory")
if( OPENMM_DIR )
	link_directories( "${OPENMM_DIR}/lib/" )
	include_directories( "${OPENMM_DIR}/include/" )
endif()

find_library( OPENMM_SERIALIAZE_LIB "OpenMMSerialization" HINT "${OPENMM_DIR}/lib" )
list( APPEND LIBS ${OPENMM_SERIALIAZE_LIB} )

# CPPUnit
set( CPPUNIT_DIR "" CACHE PATH "CPPUnit Install Directory" )
if( CPPUNIT_DIR )
	link_directories( "${CPPUNIT_DIR}/lib" )
	include_directories( "${CPPUNIT_DIR}/include" )
endif( CPPUNIT_DIR )

find_library( CPPUNIT_LIB "cppunit" hint "${CPPUNIT_DIR}/lib" )
list( APPEND LIBS ${CPPUNIT_LIB} )

# Build
add_executable( LTMDTest main.cpp ${TEST_SOURCES} ${TEST_HEADERS} )
target_link_libraries( LTMDTest "OpenMMLTMD" ${LIBS} )

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
