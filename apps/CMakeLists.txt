# Copyright (c) BBP/EPFL 2014-2015, Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

set(VOXELIZEVSD_HEADERS)
set(VOXELIZEVSD_SOURCES voxelizeVSD.cpp)
set(VOXELIZEVSD_LINK_LIBRARIES Fivox ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_application(voxelizeVSD)
