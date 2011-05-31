
# NML_TEST_PLUGIN_DIR should have been created in top-level CMakeLists.txt
if(NOT NML_TEST_PLUGIN_DIR)
    message("ERROR: plugin test folder does not exist")
endif()

file(GLOB test_sources "Test*.cpp")
foreach(test_source ${test_sources})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} ${OpenMM_LIBRARIES})

    add_test(${test_name} "${EXECUTABLE_OUTPUT_PATH}/${test_name}")

    # Environment variable works only with CMake 2.8 or higher
    set_tests_properties(${test_name}  PROPERTIES 
        ENVIRONMENT "OPENMM_PLUGIN_DIR=${NML_TEST_PLUGIN_DIR}")

endforeach(test_source ${test_sources})

