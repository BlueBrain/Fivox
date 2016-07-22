# Copyright (c) BBP/EPFL 2016, Jafet.VillafrancaDiaz@epfl.ch
# All rights reserved. Do not distribute without further notice.

set(SAMPLE-POINT_HEADERS
  ../commandLineApplication.h
)
set(SAMPLE-POINT_SOURCES
  sample-point.cpp
)
set(SAMPLE-POINT_LINK_LIBRARIES Fivox ${Boost_PROGRAM_OPTIONS_LIBRARY})

common_application(sample-point)
install(PROGRAMS plot2D.py DESTINATION bin)
