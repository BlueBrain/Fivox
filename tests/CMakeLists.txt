# Copyright (c) BBP/EPFL 2011-2015, Stefan.Eilemann@epfl.ch
# Change this number when adding tests to force a CMake run: 1

include(InstallFiles)

include_directories(${CMAKE_SOURCE_DIR}) # some tests need private headers
set(TEST_LIBRARIES ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${Boost_SYSTEM_LIBRARY} Fivox)
if(BBPSDK_FOUND)
  list(APPEND TEST_LIBRARIES ${BBPSDK_LIBRARIES})
endif()
if(Monsteer_FOUND)
  list(APPEND TEST_LIBRARIES Monsteer BrionMonsteerSpikeReport)
endif()

set(TESTDATA_TESTS sources.cpp)
if(BBPTESTDATA_FOUND AND BBPSDK_FOUND)
  set(UNIT_AND_PERF_TESTS ${TESTDATA_TESTS})
  list(APPEND TEST_LIBRARIES BBPTestData)
else()
  set(EXCLUDE_FROM_TESTS ${TESTDATA_TESTS})
endif()

include(CommonCTest)
install_files(share/Fivox/tests FILES ${TEST_FILES} COMPONENT examples)
