# Copyright (c) BBP/EPFL 2011-2015, Stefan.Eilemann@epfl.ch
# Change this number when adding tests to force a CMake run: 1

include(InstallFiles)

include_directories(${CMAKE_SOURCE_DIR}) # some tests need private headers
set(TEST_LIBRARIES ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${Boost_SYSTEM_LIBRARY} Fivox)
if(BBPSDK_FOUND)
  list(APPEND TEST_LIBRARIES ${BBPSDK_LIBRARIES})
endif()

if(BBPTESTDATA_FOUND AND BBPSDK_FOUND)
  set(UNIT_AND_PERF_TESTS compartmentSource.cpp somaSource.cpp)
else()
  set(EXCLUDE_FROM_TESTS compartmentSource.cpp somaSource.cpp)
endif()

include(CommonCTest)
install_files(share/Fivox/tests FILES ${TEST_FILES} COMPONENT examples)
