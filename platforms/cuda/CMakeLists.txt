
#########################################
### Cuda version of NML OpenMM plugin ###
#########################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/include")
include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/src")
include_directories(${CUDA_INCLUDE})

link_directories(${CUDA_TARGET_LINK})
cuda_include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/include")
cuda_include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/src")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu")
set(cuda_nml_headers ${nml_headers} ${headers})
set(cuda_nml_sources ${sources})

########################################
### Create NML plugin shared library ###
########################################

set(NML_CUDA_PLUGIN NormalModeLangevinCuda)
# On Unix or Cygwin we have to add the debug suffix manually
if(UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
    set(NML_CUDA_PLUGIN ${NML_CUDA_PLUGIN}_d)
endif()
cuda_add_library(${NML_CUDA_PLUGIN} SHARED ${cuda_nml_sources} ${cuda_nml_headers})
set_target_properties(${NML_CUDA_PLUGIN} PROPERTIES COMPILE_FLAGS "-DNMLOPENMM_CUDA_BUILDING_SHARED_LIBRARY -DNMLOPENMM_BUILDING_SHARED_LIBRARY")
target_link_libraries( ${NML_CUDA_PLUGIN} ${OpenMM_LIBRARIES} )

find_library(OpenMMCudaPlugin OpenMMCuda
    PATHS "${OpenMM_INSTALL_DIR}/lib/plugins/")
target_link_libraries(${NML_CUDA_PLUGIN} "${OpenMMCudaPlugin}")


###############
### TESTING ###
###############

if(NML_BUILD_TESTS)
    # NML_TEST_PLUGIN_DIR should have been created in top CMakeLists.txt
    if(NOT NML_TEST_PLUGIN_DIR)
        message("ERROR: plugin test folder does not exist")
    endif()
    # Copy plugin to NML_TEST_PLUGIN_DIR
    # On Windows we need to copy the correct Release/Debug plugin for testing
    if(MSVC)
        add_custom_command(TARGET ${NML_CUDA_PLUGIN} POST_BUILD
            DEPENDS $(TargetPath)
            COMMAND "${CMAKE_COMMAND}"
            ARGS -E copy \"$\(TargetPath\)\" \"${NML_TEST_PLUGIN_DIR}\"
            COMMENT "Copying normal mode langevin plugin for testing (WIN32)")
    else(MSVC)
        get_target_property(old_loc ${NML_CUDA_PLUGIN} LOCATION)
        add_custom_command(TARGET ${NML_CUDA_PLUGIN} POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
            ARGS -E copy ${old_loc} ${NML_TEST_PLUGIN_DIR}
            COMMENT "Copying normal mode langevin plugin for testing")
    endif(MSVC)
    add_subdirectory(test)
endif()

####################
### INSTALLATION ###
####################

install(TARGETS ${NML_CUDA_PLUGIN} LIBRARY DESTINATION lib/plugins)
