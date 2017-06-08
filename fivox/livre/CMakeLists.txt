# Copyright (c) 2014-2015, EPFL/Blue Brain Project
#                          Stefan.Eilemann@epfl.ch
#
# This file is part of Fivox <https://github.com/BlueBrain/Fivox>
#

if(NOT TARGET LivreData)
  return()
endif()

set(LIVREFIVOXSOURCE_HEADERS dataSource.h)
set(LIVREFIVOXSOURCE_SOURCES dataSource.cpp)
set(LIVREFIVOXSOURCE_LINK_LIBRARIES LivreData Fivox)
set(LIVREFIVOXSOURCE_OMIT_EXPORT ON)
set(LIVREFIVOXSOURCE_INCLUDE_NAME fivox/livre)

common_library(LivreFivoxSource)
