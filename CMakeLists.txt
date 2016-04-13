# Copyright (c) BBP/EPFL 2014-2016
#               Stefan.Eilemann@epfl.ch
#
# This file is part of Fivox <https://github.com/BlueBrain/Fivox>
#

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Fivox)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
  ${CMAKE_SOURCE_DIR}/CMake/common)

include(GitExternal)
include(ITKSettings)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "4")
set(VERSION_PATCH "0")
set(VERSION_ABI 4)

set(FIVOX_DESCRIPTION "ITK library to sample events into regular volumes")
set(FIVOX_MAINTAINER
  "Blue Brain Project <bbp-open-source@googlegroups.com>")
set(FIVOX_LICENSE LGPL)

set(COMMON_PROJECT_DOMAIN ch.epfl.bluebrain)
include(Common)

set(FIVOX_DEPENDENT_LIBRARIES Lunchbox ITK vmmlib)

common_package(BBPTestData)
common_package(Boost REQUIRED COMPONENTS unit_test_framework
                                         program_options system)
common_package(Brion REQUIRED)
common_package(ITK REQUIRED SYSTEM)
common_package(Livre)
common_package(Monsteer REQUIRED)
common_package(vmmlib REQUIRED)
common_package(zeq)
common_package(ZeroBuf)
common_package_post()

include(${ITK_USE_FILE})
# eile: For some unfathomable reason, the list of libraries contains
# these tokens which cause the subsequent lib target to disappear from
# the link line. Removing the tokens fixes the build, although I have
# not fully understood the issue.
list(REMOVE_ITEM ITK_LIBRARIES optimized debug)

# Try cloning lfpFivox into the 'fivox' folder
set(FIVOXLFP_DIR ${PROJECT_SOURCE_DIR}/fivox/lfp)
git_external(${FIVOXLFP_DIR} ssh://bbpcode.epfl.ch/viz/lfpFivox 2715344
             OPTIONAL)

# Add LFP-related headers and definitions if found
if(EXISTS ${FIVOXLFP_DIR}/lfpFunctor.h)
  set(FIVOX_USE_LFP ON)
  add_definitions(-DFIVOX_USE_LFP)
endif()

add_subdirectory(fivox)
add_subdirectory(apps)
add_subdirectory(tests)

include(CPackConfig)
set(DOXYGEN_MAINPAGE_MD README.md)
set(DOXYGEN_EXTRA_INPUT ${PROJECT_SOURCE_DIR}/README.md)
include(DoxygenRule)
