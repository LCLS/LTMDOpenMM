include_directories( include ../include )

set( TEST_HEADERS "include/AnalysisTest.h" "include/MathTest.h" )
set( TEST_SOURCES "src/AnalysisTest.cpp" "src/MathTest.cpp" )

# CPPUnit
set( CPPUNIT_DIR "" CACHE PATH "CPPUnit Install Directory" )
if( CPPUNIT_DIR )
	link_directories( "${CPPUNIT_DIR}/lib" )
	include_directories( "${CPPUNIT_DIR}/include" )
endif( CPPUNIT_DIR )

find_library( CPPUNIT_LIB "cppunit" hint "${CPPUNIT_DIR}/lib" )
list( APPEND LIBS ${CPPUNIT_LIB} )

# Build
if( BUILD_GPU_CUDA )
	cuda_add_executable( LTMDTest main.cpp ${TEST_SOURCES} ${TEST_HEADERS} ${CUDA_KERNELS} )
else( BUILD_GPU_CUDA )
	add_executable( LTMDTest main.cpp ${TEST_SOURCES} ${TEST_HEADERS} )
endif( BUILD_GPU_CUDA )

# Link
target_link_libraries( LTMDTest "OpenMMLTMD" ${LIBS})

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
