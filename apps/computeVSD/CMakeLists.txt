# Copyright (c) BBP/EPFL 2014-2016, Jafet.VillafrancaDiaz@epfl.ch
#
# This file is part of Fivox <https://github.com/BlueBrain/Fivox>

set(COMPUTE-VSD_HEADERS
  beerLambertProjectionImageFilter.h
  ../commandLineApplication.h
)
set(COMPUTE-VSD_SOURCES
  computeVSD.cpp
)
set(COMPUTE-VSD_LINK_LIBRARIES Fivox ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_application(compute-vsd)
