project( LTMDOpenMM )

cmake_minimum_required(VERSION 2.8)

# OpenMM
set( OPENMM_DIR "" CACHE PATH "OpenMM Install Directory")
if( OPENMM_DIR )
	link_directories( "${OPENMM_DIR}/lib/" )
	include_directories( "${OPENMM_DIR}/include/" )
endif()

set( OPENMM_SOURCE_DIR "" CACHE PATH "OpenMM Source Directory")
if( OPENMM_SOURCE_DIR )
	# JAMA
	include_directories( "${OPENMM_SOURCE_DIR}/libraries/jama/include/" )

	# SFMT
	include_directories( "${OPENMM_SOURCE_DIR}/libraries/sfmt/include/" )

	# Reference
	include_directories( "${OPENMM_SOURCE_DIR}/platforms/reference/src/" )
	include_directories( "${OPENMM_SOURCE_DIR}/platforms/reference/include/" )
endif()

find_library( OPENMM_LIB "OpenMM" HINT ${OPENMM_DIR}/lib )

include_directories( "include" )

# LTMD Library
file( GLOB LTMD_HEADERS include/LTMD/*.h )
source_group( "include\\LTMD" FILES ${LTMD_HEADERS} )

file( GLOB LTMD_SOURCES src/LTMD/*.cpp )
source_group( "src\\LTMD" FILES ${LTMD_SOURCES} )

add_library( OpenMMLTMD STATIC ${LTMD_HEADERS} ${LTMD_SOURCES})
target_link_libraries( OpenMMLTMD ${OPENMM_LIB} )

# LTMD Reference Plugin
file( GLOB LTMD_REFERENCE_HEADERS include/LTMD/Reference/*.h )
source_group( "include\\LTMD\\Reference" FILES ${LTMD_REFERENCE_HEADERS} )

file( GLOB LTMD_REFERENCE_SOURCES src/LTMD/Reference/*.cpp )
source_group( "src\\LTMD\\Reference" FILES ${LTMD_REFERENCE_SOURCES} )

add_library( LTMDReference SHARED ${LTMD_REFERENCE_HEADERS} ${LTMD_REFERENCE_SOURCES})
set_target_properties( LTMDReference PROPERTIES COMPILE_FLAGS "-DNMLOPENMM_REFERENCE_BUILDING_SHARED_LIBRARY -DNMLOPENMM_BUILDING_SHARED_LIBRARY")
target_link_libraries( LTMDReference OpenMMLTMD )

# Testing
option( BUILD_TESTING "Build test code" Off )
if( BUILD_TESTING )
	add_subdirectory( "test" )
endif( BUILD_TESTING )

# Installation
install(
	DIRECTORY "include/"
	DESTINATION "include/"
	PATTERN "*.h"
	PATTERN "include/LTMD/CUDA" EXCLUDE
	PATTERN "include/LTMD/OpenCL" EXCLUDE
	PATTERN "include/LTMD/Reference" EXCLUDE
)

install( TARGETS OpenMMLTMD LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib" )
install( TARGETS LTMDReference LIBRARY DESTINATION "${OPENMM_DIR}/lib/plugins" )
