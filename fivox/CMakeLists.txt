# Copyright (c) BBP/EPFL 2014-2015
#                        Stefan.Eilemann@epfl.ch
# All rights reserved. Do not distribute without further notice.

source_group(\\ FILES CMakeLists.txt)

set(FIVOX_PUBLIC_HEADERS
  attenuationCurve.h
  compartmentLoader.h
  densityFunctor.h
  event.h
  eventFunctor.h
  eventSource.h
  fieldFunctor.h
  frequencyFunctor.h
  imageSource.h
  imageSource.hxx
  itk.h
  somaLoader.h
  spikeLoader.h
  synapseLoader.h
  types.h
  uriHandler.h
  vsdLoader.h
)

set(FIVOX_HEADERS)

set(FIVOX_SOURCES
  eventSource.cpp
  uriHandler.cpp)

set(FIVOX_LINK_LIBRARIES
  PUBLIC ${ITK_LIBRARIES} Lunchbox vmmlib
  PRIVATE ${Boost_LIBRARIES})

# Optionally enable rtree from boost_geometry, if found
if(EXISTS "${Boost_INCLUDE_DIR}/boost/geometry.hpp")
  add_definitions(-DUSE_BOOST_GEOMETRY)
endif()

if(BBPSDK_FOUND)
  list(APPEND FIVOX_SOURCES
    compartmentLoader.cpp
    somaLoader.cpp
    synapseLoader.cpp
    vsdLoader.cpp)
  list(APPEND FIVOX_LINK_LIBRARIES BBPSDK Brain)
endif()

if(BBPTESTDATA_FOUND)
  list(APPEND FIVOX_LINK_LIBRARIES BBPTestData)
endif()

if(MONSTEER_FOUND)
  list(APPEND FIVOX_SOURCES
    spikeLoader.cpp )
  list(APPEND FIVOX_LINK_LIBRARIES Monsteer)
endif()

common_library(Fivox)
add_subdirectory(livre)
