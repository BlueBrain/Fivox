# Copyright (c) BBP/EPFL 2014-2016
#               Stefan.Eilemann@epfl.ch
#
# This file is part of Fivox <https://github.com/BlueBrain/Fivox>
#

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Fivox VERSION 0.6.0)
set(Fivox_VERSION_ABI 6)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
                              ${CMAKE_SOURCE_DIR}/CMake/common)

include(GitExternal)
include(ITKSettings)

set(FIVOX_DESCRIPTION "ITK library to sample events into regular volumes")
set(FIVOX_MAINTAINER
  "Blue Brain Project <bbp-open-source@googlegroups.com>")
set(FIVOX_LICENSE LGPL)
set(FIVOXLFP_DIR ${PROJECT_SOURCE_DIR}/fivox/lfp)

set(COMMON_PROJECT_DOMAIN ch.epfl.bluebrain)
include(Common)

set(FIVOX_DEPENDENT_LIBRARIES CUDA Lunchbox ITK vmmlib)

common_find_package(BBPTestData)
common_find_package(Boost REQUIRED COMPONENTS unit_test_framework
                                              program_options system)
common_find_package(Brion REQUIRED)
#CUDA is currently disabled because there is a bug.
#common_find_package(CUDA 4.0)
common_find_package(ITK REQUIRED SYSTEM)
common_find_package(Lexis)
common_find_package(Livre)
common_find_package(vmmlib REQUIRED)
common_find_package(ZeroEQ)

if(CUDA_FOUND)
  option(FIVOX_USE_CUDA "Use CUDA" ON)
  if(NOT FIVOX_USE_CUDA)
    common_find_package_disable(CUDA)
  endif()
endif()

common_find_package_post()

include(${ITK_USE_FILE})
# eile: For some unfathomable reason, the list of libraries contains
# these tokens which cause the subsequent lib target to disappear from
# the link line. Removing the tokens fixes the build, although I have
# not fully understood the issue.
list(REMOVE_ITEM ITK_LIBRARIES optimized debug)

add_subdirectory(fivox)
add_subdirectory(apps)
add_subdirectory(tests)

include(CPackConfig)
set(DOXYGEN_MAINPAGE_MD README.md)
set(DOXYGEN_EXTRA_INPUT ${PROJECT_SOURCE_DIR}/README.md)
include(DoxygenRule)
