# Copyright (c) BBP/EPFL 2014-2015, Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

set(COMPUTE-VSD_HEADERS
  beerLambertProjectionImageFilter.h
  ../commandLineApplication.h
)
set(COMPUTE-VSD_SOURCES
  computeVSD.cpp
)
set(COMPUTE-VSD_LINK_LIBRARIES Fivox ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_application(compute-vsd)
