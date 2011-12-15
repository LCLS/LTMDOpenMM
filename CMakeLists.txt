project( LTMDOpenMM )
cmake_minimum_required(VERSION 2.8)

include_directories( "include" )

# Profiling
option( BUILD_PROFILE "Build Profiling code" Off )
if( BUILD_PROFILE )
	option( BUILD_PROFILE_INTEGRATOR "Build Integrator Profiling code" Off )
	if( BUILD_PROFILE_INTEGRATOR )
		add_definitions( "-DPROFILE_INTEGRATOR" )
	endif( BUILD_PROFILE_INTEGRATOR )
	
	option( BUILD_PROFILE_ANALYSIS "Build Analysis Profiling code" Off )
	if( BUILD_PROFILE_ANALYSIS )
		add_definitions( "-DPROFILE_ANALYSIS" )
	endif( BUILD_PROFILE_ANALYSIS )

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -pg")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif( BUILD_PROFILE )

# Profiling
option( BUILD_VALIDATION "Build Validation Code" Off )
if( BUILD_VALIDATION )
	add_definitions( "-DVALIDATION" )
endif( BUILD_VALIDATION )

# Intel
if (${CMAKE_CXX_COMPILER} MATCHES "icc.*$") 
	set( CUDA_NVCC_FLAGS "-ccbin=icc" )
endif()

# MKL
option( BUILD_MKL "Build with Intel MKL code" Off )
if( BUILD_MKL )
	add_definitions( "-DINTEL_MKL" )

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl=parallel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -mkl=parallel")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mkl=parallel")
endif( BUILD_MKL )

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

	# CUDA
	include_directories( "${OPENMM_SOURCE_DIR}/platforms/cuda/src/" )
	include_directories( "${OPENMM_SOURCE_DIR}/platforms/cuda/include/" )
endif()

find_library( OPENMM_LIB "OpenMM" HINT ${OPENMM_DIR}/lib )

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

option( BUILD_GPU "Build GPU code" Off )
if( BUILD_GPU )
	option( BUILD_GPU_CUDA "Build Cuda GPU code" Off )
	if( BUILD_GPU_CUDA )
		find_package( CUDA REQUIRED )
		find_library( OPENMM_CUDA_LIB "OpenMMCuda" HINT "${OPENMM_DIR}/lib/plugins/")

		file( GLOB LTMD_CUDA_HEADERS include/LTMD/CUDA/*.h )
		source_group( "include\\LTMD\\CUDA" FILES ${LTMD_CUDA_HEADERS} )

		file( GLOB LTMD_CUDA_SOURCES src/LTMD/CUDA/*.cpp )
		source_group( "src\\LTMD\\CUDA" FILES ${LTMD_CUDA_SOURCES} )

		file( GLOB LTMD_CUDA_KERNELS src/LTMD/CUDA/kernels/*.cu )
		source_group( "Kernels\\LTMD" FILES ${LTMD_CUDA_KERNELS} )

		CUDA_ADD_LIBRARY( LTMDCuda SHARED ${LTMD_CUDA_HEADERS} ${LTMD_CUDA_SOURCES} ${LTMD_CUDA_KERNELS} )
		set_target_properties( LTMDCuda PROPERTIES COMPILE_FLAGS "-DNMLOPENMM_CUDA_BUILDING_SHARED_LIBRARY -DNMLOPENMM_BUILDING_SHARED_LIBRARY")
		target_link_libraries( LTMDCuda OpenMMLTMD ${OPENMM_CUDA_LIB} )

		install( TARGETS LTMDCuda LIBRARY DESTINATION "${OPENMM_DIR}/lib/plugins" )
	endif( BUILD_GPU_CUDA )

	option( BUILD_GPU_OPENCL "Build OpenCL GPU code" Off )
	if( BUILD_GPU_OPENCL )
		file( GLOB LTMD_OPENCL_HEADERS include/LTMD/OpenCL/*.h )
		source_group( "include\\LTMD\\OpenCL" FILES ${LTMD_OPENCL_HEADERS} )

		file( GLOB LTMD_OPENCL_SOURCES src/LTMD/OpenCL/*.cpp )
		source_group( "src\\LTMD\\OpenCL" FILES ${LTMD_OPENCL_SOURCES} )

		add_library( LTMDOpenCL SHARED ${LTMD_OPENCL_HEADERS} ${LTMD_OPENCL_SOURCES})
		set_target_properties( LTMDOpenCL PROPERTIES COMPILE_FLAGS "-DNMLOPENMM_REFERENCE_BUILDING_SHARED_LIBRARY -DNMLOPENMM_BUILDING_SHARED_LIBRARY")
		target_link_libraries( LTMDOpenCL OpenMMLTMD )

		install( TARGETS LTMDOpenCL LIBRARY DESTINATION "${OPENMM_DIR}/lib/plugins" )
	endif( BUILD_GPU_OPENCL )
endif( BUILD_GPU )

# Testing
option( BUILD_TESTING "Build test code" Off )
if( BUILD_TESTING )
	add_subdirectory( "test" )
endif( BUILD_TESTING )

# Installation
install(
	DIRECTORY "include/"
	DESTINATION "include"
	PATTERN "*.h"
	PATTERN ".*" EXCLUDE
	PATTERN "include/LTMD/CUDA" EXCLUDE
	PATTERN "include/LTMD/OpenCL" EXCLUDE
	PATTERN "include/LTMD/Reference" EXCLUDE
)

install( TARGETS OpenMMLTMD LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib" )
install( TARGETS LTMDReference LIBRARY DESTINATION "${OPENMM_DIR}/lib/plugins" )
