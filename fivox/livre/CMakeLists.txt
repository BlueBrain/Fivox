# Copyright (c) 2014-2015, EPFL/Blue Brain Project
#                          Stefan.Eilemann@epfl.ch
#
# This file is part of Fivox <https://github.com/BlueBrain/Fivox>
#

if(NOT LIVRE_FOUND)
  return()
endif()

set(LIVREFIVOXSOURCE_HEADERS dataSource.h)
set(LIVREFIVOXSOURCE_SOURCES dataSource.cpp)
set(LIVREFIVOXSOURCE_LINK_LIBRARIES ${LIVRECORE_LIBRARY} Fivox)
set(LIVREFIVOXSOURCE_INCLUDE_NAME fivox/livre)

common_library(LivreFivoxSource)
