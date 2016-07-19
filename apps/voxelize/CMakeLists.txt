# Copyright (c) BBP/EPFL 2014-2015, Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

set(VOXELIZE_HEADERS
  beerLambertProjectionImageFilter.h
  ../commandLineApplication.h
  scaleFilter.h
  volumeHandler.h
  volumeWriter.h
)
set(VOXELIZE_SOURCES
  voxelize.cpp
)
set(VOXELIZE_LINK_LIBRARIES Fivox ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_application(voxelize)
