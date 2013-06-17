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

# Validation
option( BUILD_VALIDATION "Build Validation Code" Off )
if( BUILD_VALIDATION )
	add_definitions( "-DVALIDATION" )
endif( BUILD_VALIDATION )

option( BUILD_KERNEL_VALIDATION "Build Kernel Validation Code" Off )
if( BUILD_KERNEL_VALIDATION )
	add_definitions( "-DKERNEL_VALIDATION" )
endif( BUILD_KERNEL_VALIDATION )

# MKL
option( BUILD_MKL "Build with Intel MKL code" Off )
if( BUILD_MKL )
	add_definitions( "-DINTEL_MKL" )

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl=parallel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -mkl=parallel")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mkl=parallel")
else( BUILD_MKL )
    find_library( LIB_BLAS "blas" HINTS "${BLAS_LIBPATH}" )
    find_library( LIB_LAPACK "lapack" HINTS "${LAPACK_LIBPATH}" )

    if( NOT ${LIB_BLAS} MATCHES "LIB_BLAS-NOTFOUND" AND NOT ${LIB_LAPACK} MATCHES "LIB_LAPACK-NOTFOUND" )
        set( LIBS ${LIBS} ${LIB_BLAS} ${LIB_LAPACK} )
    else()
        if( ${LIB_BLAS} MATCHES "LIB_BLAS-NOTFOUND" )
            set( BLAS_INCPATH "" CACHE PATH "Blas include directory" )
            set( BLAS_LIBPATH "" CACHE PATH "Blas library directory" )
        endif()

        if( ${LIB_LAPACK} MATCHES "LIB_LAPACK-NOTFOUND" )
            set( LAPACK_INCPATH "" CACHE PATH "Lapack include directory" )
            set( LAPACK_LIBPATH "" CACHE PATH "Lapack library directory" )
        endif()
    endif()
endif( BUILD_MKL )

# OpenMP
option( BUILD_OPENMP "Build with openmp support" Off )
if( BUILD_OPENMP )
	# Intel 
	if( ${CMAKE_CXX_COMPILER} MATCHES "icpc.*$" OR ${CMAKE_CXX_COMPILER} MATCHES "icc.*$" ) 
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -openmp")
	else()
        # Other
        include( FindOpenMP )
        if( OPENMP_FOUND )
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
          set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        endif( OPENMP_FOUND )
    endif()
endif( BUILD_OPENMP )

# OpenMM
set( OPENMM_DIR "$ENV{OPENMM_HOME}" CACHE PATH "OpenMM Install Directory")
if( OPENMM_DIR )
	link_directories( "${OPENMM_DIR}/lib/" )
	include_directories( "${OPENMM_DIR}/include/" )
endif()

set( OPENMM_SOURCE_DIR "$ENV{OPENMM_SOURCE}" CACHE PATH "OpenMM Source Directory")
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
	include_directories( "${OPENMM_SOURCE_DIR}/libraries/lepton/include" )
endif()

find_library( OPENMM_LIB "OpenMM" HINT ${OPENMM_DIR}/lib )

set( LIBS ${OPENMM_LIB} ${LIBS} )

# First Order
option( BUILD_FIRST_ORDER "Build with first order perterb" Off )
if( BUILD_FIRST_ORDER )
	add_definitions( "-DFIRST_ORDER" )
endif( BUILD_FIRST_ORDER )

# LTMD Library
file( GLOB LTMD_HEADERS include/LTMD/*.h )
source_group( "include\\LTMD" FILES ${LTMD_HEADERS} )

file( GLOB LTMD_SOURCES src/LTMD/*.cpp )
source_group( "src\\LTMD" FILES ${LTMD_SOURCES} )

add_library( OpenMMLTMD SHARED ${LTMD_HEADERS} ${LTMD_SOURCES})
target_link_libraries( OpenMMLTMD ${LIBS} )

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
	option( BUILD_FAST_NOISE "Build Fast Noise code" On )
	if( BUILD_FAST_NOISE )
		add_definitions( "-DFAST_NOISE" )
	endif( BUILD_FAST_NOISE )

	option( BUILD_GPU_CUDA "Build Cuda GPU code" Off )
	if( BUILD_GPU_CUDA )
		find_package( CUDA REQUIRED )
		find_library( OPENMM_CUDA_LIB "OpenMMCuda" HINT "${OPENMM_DIR}/lib/plugins/")

		file( GLOB LTMD_CUDA_HEADERS include/LTMD/CUDA/*.h  )
		source_group( "include\\LTMD\\CUDA" FILES ${LTMD_CUDA_HEADERS} )
	        include_directories( "include/LTMD/CUDA/" )

		file( GLOB LTMD_CUDA_SOURCES src/LTMD/CUDA/*.cpp )
		source_group( "src\\LTMD\\CUDA" FILES ${LTMD_CUDA_SOURCES} )

		file( GLOB LTMD_CUDA_KERNELS src/LTMD/CUDA/kernels/*.cpp )
		source_group( "Kernels\\LTMD" FILES ${LTMD_CUDA_KERNELS} )
		
		# Fast Math
		option( BUILD_GPU_CUDA_FAST_MATH "Build with fast math" Off )
		if( BUILD_GPU_CUDA_FAST_MATH )
			list( APPEND CUDA_NVCC_FLAGS "-use_fast_math" )
		endif( BUILD_GPU_CUDA_FAST_MATH )
		
		# Intel
		if( ${CMAKE_CXX_COMPILER} MATCHES "icpc.*$" OR ${CMAKE_CXX_COMPILER} MATCHES "icc.*$" ) 
			list( APPEND CUDA_NVCC_FLAGS "-ccbin=icc" )
		endif()

		CUDA_ADD_LIBRARY( LTMDCuda SHARED ${LTMD_CUDA_HEADERS} ${LTMD_CUDA_SOURCES} ${LTMD_CUDA_KERNELS} )
		set_target_properties( LTMDCuda PROPERTIES COMPILE_FLAGS "-DNMLOPENMM_CUDA_BUILDING_SHARED_LIBRARY -DNMLOPENMM_BUILDING_SHARED_LIBRARY")
		target_link_libraries( LTMDCuda OpenMMLTMD ${OPENMM_CUDA_LIB} )

		install( TARGETS LTMDCuda LIBRARY DESTINATION "lib/plugins" )
	endif( BUILD_GPU_CUDA )
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
	PATTERN "include/LTMD/Reference" EXCLUDE
)

install( TARGETS OpenMMLTMD LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib" )
install( TARGETS LTMDReference LIBRARY DESTINATION "lib/plugins" )
