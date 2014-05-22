# Copyright (c) BBP/EPFL 2011-2014
#                        Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

source_group(\\ FILES CMakeLists.txt)

set(FIVOX_PUBLIC_HEADERS ${COMMON_INCLUDES} world.h)
set(FIVOX_HEADERS)
set(FIVOX_SOURCES ${COMMON_SOURCES} world.cpp)
set(FIVOX_LINK_LIBRARIES ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_library(Fivox)
