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

####################################
### REFERENCE CPU IMPLEMENTATION ###
####################################

include_directories("${OPENMM_SOURCE_DIR}/platforms/reference/include")
include_directories("${OPENMM_SOURCE_DIR}/platforms/reference/src")
include_directories("${OPENMM_SOURCE_DIR}/libraries/sfmt/include")

# Reference platform implementation
include_directories(${CMAKE_SOURCE_DIR}/platforms/reference/include)
include_directories(${CMAKE_SOURCE_DIR}/platforms/reference/src)
file(GLOB headers "${CMAKE_SOURCE_DIR}/platforms/reference/include/*.h")
file(GLOB sources "${CMAKE_SOURCE_DIR}/platforms/reference/src/*.cpp")
set(nml_headers ${nml_headers} ${headers})
set(nml_sources ${nml_sources} ${sources})

########################################
### Create NML plugin shared library ###
########################################

add_library(${NML_PLUGIN} SHARED ${nml_sources} ${nml_headers})
set_target_properties(${NML_PLUGIN} PROPERTIES COMPILE_FLAGS "-DNMLOPENMM_REFERENCE_BUILDING_SHARED_LIBRARY -DNMLOPENMM_BUILDING_SHARED_LIBRARY")
target_link_libraries( ${NML_PLUGIN} ${OpenMM_LIBRARIES})

###############
### TESTING ###
###############

if(NML_BUILD_TESTS)
    # NML_TEST_PLUGIN_DIR should have been created in top CMakeLists.txt
    if(NOT NML_TEST_PLUGIN_DIR)
        message("ERROR: plugin test folder does not exist ${NML_TEST_PLUGIN_DIR}")
    endif()
    # Copy plugin to NML_TEST_PLUGIN_DIR
    # On Windows we need to copy the correct Release/Debug plugin for testing
    if(MSVC)
        add_custom_command(TARGET ${NML_PLUGIN} POST_BUILD
            DEPENDS $(TargetPath)
            COMMAND "${CMAKE_COMMAND}"
            ARGS -E copy \"$\(TargetPath\)\" \"${NML_TEST_PLUGIN_DIR}\"
            COMMENT "Copying normal mode langevin plugin for testing (WIN32)")
    else(MSVC)
        get_target_property(old_loc ${NML_PLUGIN} LOCATION)
        add_custom_command(TARGET ${NML_PLUGIN} POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
            ARGS -E copy ${old_loc} ${NML_TEST_PLUGIN_DIR}
            COMMENT "Copying normal mode langevin plugin for testing")
    endif(MSVC)
    add_subdirectory(test)
endif()

####################
### INSTALLATION ###
####################

if(WIN32)
    # install DLL but not LIB
    install(TARGETS ${NML_PLUGIN} RUNTIME DESTINATION lib)
else(WIN32)
    install(TARGETS ${NML_PLUGIN} LIBRARY DESTINATION lib)
endif(WIN32)
