#/* -------------------------------------------------------------------------- *
# *                                   OpenMM                                   *
# * -------------------------------------------------------------------------- *
# * This is part of the OpenMM molecular simulation toolkit originating from   *
# * Simbios, the NIH National Center for Physics-Based Simulation of           *
# * Biological Structures at Stanford, funded under the NIH Roadmap for        *
# * Medical Research, grant U54 GM072970. See https://simtk.org.               *
# *                                                                            *
# * Portions copyright (c) 2009 Stanford University and the Authors.           *
# * Authors: Chris Sweet, Christopher Bruns                                    *
# * Contributors:                                                              *
# *                                                                            *
# * Permission is hereby granted, free of charge, to any person obtaining a    *
# * copy of this software and associated documentation files (the "Software"), *
# * to deal in the Software without restriction, including without limitation  *
# * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
# * and/or sell copies of the Software, and to permit persons to whom the      *
# * Software is furnished to do so, subject to the following conditions:       *
# *                                                                            *
# * The above copyright notice and this permission notice shall be included in *
# * all copies or substantial portions of the Software.                        *
# *                                                                            *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
# * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
# * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
# * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
# * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
# * -------------------------------------------------------------------------- */

# Create Normal Mode Langevin (NML) plugin for OpenMM
project(NMLOpenMM)

# Only cmake 2.8 has FindCuda.cmake
cmake_minimum_required(VERSION 2.8)

# Enable nightly builds
include(Dart)

# Dart server is only cmake 2.4, so don't let it run this whole file
# Why does kitware think it's cool to validate the whole CMake hierarchy
# on the dart server?
set(cmv "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}")
if(cmv EQUAL "2.4")
    set(IS_DART_SERVER_SO_IGNORE_MOST_CMAKE TRUE)
else(cmv EQUAL "2.4")
    set(IS_DART_SERVER_SO_IGNORE_MOST_CMAKE FALSE)
endif(cmv EQUAL "2.4")


if(NOT IS_DART_SERVER_SO_IGNORE_MOST_CMAKE)

# We have a custom cmake module for FindOpenMM
# to find existing external install of OpenMM (required)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules" "${OPENMM_SOURCE_DIR}/cmake_modules")
# ...OpenMM
find_package(OpenMM REQUIRED)
mark_as_advanced(CLEAR OpenMM_INCLUDE_DIR OpenMM_LIBRARIES)

# Build 32 bit binaries, since CUDA doesn't currently work with 64 bit
IF (APPLE)
   SET (CMAKE_CXX_FLAGS "-arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5")
   SET (CMAKE_C_FLAGS "-arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5")
ENDIF (APPLE)

# Put local includes before installed ones
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${OpenMM_INCLUDE_DIR}")
include_directories("${OPENMM_SOURCE_DIR}/libraries/jama/include")

# This is an openmm plugin.  It should install into the OpenMM install area.
# Deduce OpenMM install path from the include directory populated by find_package(OpenMM)
find_path(OpenMM_INSTALL_DIR
    include
    PATHS "${OpenMM_INCLUDE_DIR}/.."
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
    "${OpenMM_INSTALL_DIR}" CACHE PATH "OpenMM plugin install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB nml_headers "include/*.h")
file(GLOB nml_sources "src/*.cpp")


# The NML platforms need access to the OpenMM source tree headers
# which are not installed by default
file(TO_CMAKE_PATH "$ENV{USERPROFILE}" USERPROFILE)
find_path(OPENMM_SOURCE_DIR
    platforms/reference/include/ReferencePlatform.h
    PATHS
        "$ENV{HOME}/svn/OpenMM"
        "${USERPROFILE}/svn/OpenMM"
        "${USERPROFILE}/My Documents/svn/OpenMM"
        "C:/simtk/work-src/openmm/trunk"
        "$ENV{HOME}/simtk/work-src/openmm/trunk"
    PATH_SUFFIXES OpenMM
)

####################################
### Reference CPU implementation ###
####################################

set(NML_PLUGIN NormalModeLangevin)
# On Unix or Cygwin we have to add the debug suffix manually
if(UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
    set(NML_PLUGIN ${NML_PLUGIN}_d)
endif()
add_subdirectory(platforms/reference)


######################################
### CUDA NVIDIA GPU IMPLEMENTATION ###
######################################

SET(FINDCUDA_DIR "${OPENMM_SOURCE_DIR}/cmake_modules/FindCUDA")
SET(CUDA_BUILD_CUBIN OFF)
FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    IF(NOT CUDA_NVCC_FLAGS)
        SET(FLAGS "")
        # Note that cmake will insert semicolons between these item automatically...
        SET(FLAGS ${FLAGS} -gencode arch=compute_11,code=sm_11)
        SET(FLAGS ${FLAGS} -gencode arch=compute_13,code=sm_13)
        SET(FLAGS ${FLAGS} -gencode arch=compute_20,code=sm_20)
        SET(FLAGS ${FLAGS} -use_fast_math)
        IF(MSVC)
            # Unfortunately the variables CUDA_NVCC_FLAGS_RELEASE and CUDA_NVCC_FLAGS_DEBUG
            # appear to be unused, at least in CMake 2.6
            # Release /MD linkage
            set(FLAGS ${FLAGS} "-Xcompiler \"/MD\"")
        ENDIF(MSVC)
        SET(CUDA_NVCC_FLAGS  "${FLAGS}"
            CACHE STRING "Semicolon delimit multiple arguments")
    ENDIF(NOT CUDA_NVCC_FLAGS)
    SET(OPENMM_BUILD_CUDA_LIB ON CACHE BOOL "Build OpenMMCuda library for Nvidia GPUs")

    # I wish I was not hardcoding /MD here
    # Avoid strange windows link error with cuda free energy
    # plugin by specifying /MD to CUDA_NVCC_FLAGS
    MARK_AS_ADVANCED(CLEAR CUDA_NVCC_FLAGS)
    IF(MSVC)
        # Unfortunately the variables CUDA_NVCC_FLAGS_RELEASE and CUDA_NVCC_FLAGS_DEBUG
        # appear to be unused, at least in CMake 2.6
        # Release /MD linkage
        SET(HAS_NVCC_FLAG FALSE)
        IF(CUDA_NVCC_FLAGS MATCHES "-Xcompiler")
            SET(HAS_NVCC_FLAG TRUE)
        ENDIF(CUDA_NVCC_FLAGS MATCHES "-Xcompiler")
        IF(NOT HAS_NVCC_FLAG)
            set(new_flags "-Xcompiler \"/MD\"")
            if(CUDA_NVCC_FLAGS)
                set(new_flags "${CUDA_NVCC_FLAGS};${new_flags}")
            endif(CUDA_NVCC_FLAGS)
            SET(CUDA_NVCC_FLAGS "${new_flags}"
                CACHE STRING "Semicolon delimit multiple arguments" FORCE)
        ENDIF(NOT HAS_NVCC_FLAG)
    ENDIF(MSVC)
ELSE(CUDA_FOUND)
    SET(OPENMM_BUILD_CUDA_LIB OFF CACHE BOOL "Build OpenMMCuda library for Nvidia GPUs")
ENDIF(CUDA_FOUND)
IF(OPENMM_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(OPENMM_BUILD_CUDA_LIB)
MARK_AS_ADVANCED(CUDA_VERBOSE_BUILD)
MARK_AS_ADVANCED(CUDA_BUILD_CUBIN)
MARK_AS_ADVANCED(CUDA_BUILD_EMULATION)

set(NML_BUILD_TESTS ON CACHE BOOL "Whether to build test programs")
if(NML_BUILD_TESTS)
    enable_testing()

    # Create directory for testing plugins
    set(NML_TEST_PLUGIN_DIR "${CMAKE_BINARY_DIR}/test_plugin_dir" CACHE PATH "Where to store plugins for testing")
    mark_as_advanced(NML_TEST_PLUGIN_DIR)
    file(MAKE_DIRECTORY "${NML_TEST_PLUGIN_DIR}")
    # Copy installed OpenMM plugins into test install dir.  Especially Cuda plugin
    file(GLOB OPENMM_PLUGIN_FILES "${OpenMM_INSTALL_DIR}/lib/plugins/*.*")
    # message("${OPENMM_PLUGIN_FILES}")
    foreach(PLUGIN ${OPENMM_PLUGIN_FILES})
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E copy "${PLUGIN}" "${NML_TEST_PLUGIN_DIR}")
    endforeach(PLUGIN ${OPENMM_PLUGIN_FILES})

    add_subdirectory(test)
endif(NML_BUILD_TESTS)

# Also install include files
FILE(GLOB TOP_HEADERS      include/nmlopenmm/*.h          */include/nmlopenmm/*.h)
install_files(/include/nmlopenmm          FILES ${TOP_HEADERS})

# Close dart server cmake ignoring hack
endif(NOT IS_DART_SERVER_SO_IGNORE_MOST_CMAKE)

