# Copyright (c) BBP/EPFL 2014
#                        Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

source_group(\\ FILES CMakeLists.txt)

set(FIVOX_PUBLIC_HEADERS
  ${COMMON_INCLUDES}
  event.h
  eventFunctor.h
  eventSource.h
  imageSource.h
  imageSource.hxx
  somaLoader.h
  types.h
)

set(FIVOX_HEADERS)
set(FIVOX_SOURCES ${COMMON_SOURCES})
set(FIVOX_LINK_LIBRARIES ${ITK_LIBRARIES})

common_library(Fivox)
