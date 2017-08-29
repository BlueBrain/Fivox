# Copyright (c) BBP/EPFL 2014-2017
#               Stefan.Eilemann@epfl.ch
#
# This file is part of Fivox <https://github.com/BlueBrain/Fivox>
#

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Fivox VERSION 0.7.0)
set(Fivox_VERSION_ABI 7)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
                              ${CMAKE_SOURCE_DIR}/CMake/common)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/CMake/common/Common.cmake)
  message(FATAL_ERROR "CMake/common missing, run: git submodule update --init")
endif()

include(GitExternal)

set(FIVOX_DESCRIPTION "ITK library to sample events into regular volumes")
set(FIVOX_MAINTAINER
  "Blue Brain Project <bbp-open-source@googlegroups.com>")
set(FIVOX_LICENSE LGPL)
set(FIVOXLFP_DIR ${PROJECT_SOURCE_DIR}/fivox/lfp)
set(FIVOX_DEP_DEPENDS libinsighttoolkit4-dev)

set(COMMON_PROJECT_DOMAIN ch.epfl.bluebrain)
include(Common)

set(FIVOX_DEPENDENT_LIBRARIES CUDA Lunchbox ITK vmmlib)

common_find_package(BBPTestData)
common_find_package(Boost REQUIRED COMPONENTS unit_test_framework
                                              program_options system)
common_find_package(Brion REQUIRED)
common_find_package(CUDA 4.0)
common_find_package(ITK REQUIRED SYSTEM)
common_find_package(Lexis)
common_find_package(Livre)
common_find_package(Monsteer) # for tests using streaming plugin
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
add_subdirectory(fivox)
add_subdirectory(apps)
add_subdirectory(tests)

include(CPackConfig)
set(DOXYGEN_MAINPAGE_MD README.md)
set(DOXYGEN_EXTRA_INPUT ${PROJECT_SOURCE_DIR}/README.md)
include(DoxygenRule)
