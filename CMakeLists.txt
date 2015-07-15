# Copyright (c) BBP/EPFL 2014-2015
#               Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Fivox)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
  ${CMAKE_SOURCE_DIR}/CMake/common)

include(GitExternal)
include(ITKSettings)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "3")
set(VERSION_PATCH "0")
set(VERSION_ABI 3)

set(FIVOX_DESCRIPTION "Library to sample fields into regular volumes")

include(Common)

common_package(BBPSDK)
common_package(BBPTestData)
common_package(Boost REQUIRED COMPONENTS unit_test_framework
                                         program_options system)
common_package(ITK REQUIRED)
common_package(Livre)
common_package(vmmlib REQUIRED)
common_package_post()

set(FIVOX_DEPENDENT_LIBRARIES ITK vmmlib BBPSDK)

include(${ITK_USE_FILE})

# eile: For some unfathomable reason, the list of libraries contains
# these tokens which cause the subsequent lib target to disappear from
# the link line. Removing the tokens fixes the build, although I have
# not fully understood the issue.
list(REMOVE_ITEM ITK_LIBRARIES optimized debug)

add_subdirectory(fivox)
add_subdirectory(apps)
add_subdirectory(tests)
add_subdirectory(configs)

set(GIT_DOCUMENTATION_REPO BBPDocumentation)
set(COMMON_ORGANIZATION_NAME BBP)
set(COMMON_PROJECT_DOMAIN ch.epfl.bluebrain)
include(CPackConfig)
set(DOXYGEN_MAINPAGE_MD README.md)
set(DOXYGEN_EXTRA_INPUT ${PROJECT_SOURCE_DIR}/README.md)
include(DoxygenRule)
