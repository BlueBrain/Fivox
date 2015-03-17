# Copyright (c) BBP/EPFL 2011-2015, Stefan.Eilemann@epfl.ch
# Change this number when adding tests to force a CMake run: 1

include(InstallFiles)

include_directories(${CMAKE_SOURCE_DIR}) # some tests need private headers
set(TEST_LIBRARIES ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${Boost_SYSTEM_LIBRARY} Fivox)
if(BBPSDK_FOUND)
  list(APPEND TEST_LIBRARIES ${BBPSDK_LIBRARIES})
endif()

if(NOT BBPTESTDATA_FOUND OR NOT BBPSDK_FOUND)
  set(EXCLUDE_FROM_TESTS compartmentSource.cpp somaSource.cpp)
endif()

include(CommonCTest)
install_files(share/Fivox/tests FILES ${TEST_FILES} COMPONENT examples)

if(BBPTESTDATA_FOUND AND BBPSDK_FOUND)
  # These can be compiled as perf and unit tests, re-add them as perf tests
  foreach(TEST compartmentSource somaSource)
    add_executable(perf_${TEST} ${TEST}.cpp)
    target_link_libraries(perf_${TEST} ${TEST_LIBRARIES})
    set_target_properties(perf_${TEST} PROPERTIES FOLDER "Tests"
      OUTPUT_NAME perf_${TEST})
    add_test(NAME perf_${TEST} COMMAND $<TARGET_FILE:perf_${TEST}>)
    add_dependencies(${PROJECT_NAME}_run_perf_tests perf_${TEST})
  endforeach()
endif()