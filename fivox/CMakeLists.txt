# Copyright (c) BBP/EPFL 2014-2016
#                        Stefan.Eilemann@epfl.ch

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
  progressObserver.h
  somaLoader.h
  spikeLoader.h
  synapseLoader.h
  testLoader.h
  types.h
  uriHandler.h
  vsdLoader.h
)

if(FIVOX_USE_LFP)
  list(APPEND FIVOX_PUBLIC_HEADERS lfp/lfpFunctor.h)
endif()

set(FIVOX_SOURCES
  compartmentLoader.cpp
  eventSource.cpp
  progressObserver.cpp
  somaLoader.cpp
  spikeLoader.cpp
  synapseLoader.cpp
  testLoader.cpp
  uriHandler.cpp
  vsdLoader.cpp
)

set(FIVOX_LINK_LIBRARIES
  PUBLIC ${ITK_LIBRARIES} Brion Lunchbox vmmlib
  PRIVATE ${Boost_LIBRARIES} Brain Monsteer)

# Optionally enable rtree from boost_geometry, if found
if(EXISTS "${Boost_INCLUDE_DIR}/boost/geometry.hpp")
  add_definitions(-DUSE_BOOST_GEOMETRY)
endif()

if(TARGET zeq AND TARGET ZeroBufData)
  add_definitions(-DUSE_ZEROEQ_PROGRESS)
  list(APPEND FIVOX_LINK_LIBRARIES PRIVATE zeq ZeroBufData)
endif()

if(BBPTESTDATA_FOUND)
  list(APPEND FIVOX_LINK_LIBRARIES PRIVATE BBPTestData)
endif()

common_library(Fivox)
add_subdirectory(livre)
