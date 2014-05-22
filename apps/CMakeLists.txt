# Copyright (c) BBP/EPFL 2014
#                        Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

set(SPHERES_HEADERS)
set(SPHERES_SOURCES spheres.cpp)
set(SPHERES_LINK_LIBRARIES Fivox ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_application(spheres)
