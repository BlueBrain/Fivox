# Copyright (c) BBP/EPFL 2014-2016
#                        Stefan.Eilemann@epfl.ch

set(FIVOX_PUBLIC_HEADERS
  attenuationCurve.h
  compartmentLoader.h
  densityFunctor.h
  eventValueSummationImageSource.h
  eventValueSummationImageSource.hxx
  functorImageSource.h
  functorImageSource.hxx
  eventFunctor.h
  eventSource.h
  fieldFunctor.h
  frequencyFunctor.h
  imageSource.h
  imageSource.hxx
  progressObserver.h
  scaleFilter.h
  somaLoader.h
  spikeLoader.h
  synapseLoader.h
  testLoader.h
  types.h
  uriHandler.h
  volumeHandler.h
  vsdLoader.h
)

# Try cloning lfpFivox into the 'fivox' folder
git_external(${FIVOXLFP_DIR} ssh://bbpcode.epfl.ch/viz/lfpFivox f1360df
             OPTIONAL)
if(EXISTS ${FIVOXLFP_DIR}/lfpFunctor.h)
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
  volumeHandler.cpp
  vsdLoader.cpp
)

set(FIVOX_LINK_LIBRARIES
  PUBLIC ${ITK_LIBRARIES} Brion Lunchbox vmmlib
  PRIVATE ${Boost_LIBRARIES} Brain)

if(FIVOX_USE_CUDA)
  add_subdirectory(cuda)
  list(APPEND PRIVATE FIVOX_LINK_LIBRARIES ${CUDA_LIBRARIES})
  list(APPEND FIVOX_PUBLIC_HEADERS cudaImageSource.h cudaImageSource.hxx)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

common_library(Fivox)

if(TARGET Lexis AND TARGET ZeroEQ)
  target_compile_definitions(Fivox PRIVATE USE_ZEROEQ_PROGRESS)
  target_link_libraries(Fivox PRIVATE Lexis ZeroEQ)
endif()

if(TARGET BBPTestData)
  target_link_libraries(Fivox PRIVATE BBPTestData)
endif()

# Optionally enable rtree from boost_geometry, if found
if(EXISTS "${Boost_INCLUDE_DIR}/boost/geometry.hpp")
  target_compile_definitions(Fivox PRIVATE USE_BOOST_GEOMETRY)
endif()
if(EXISTS ${FIVOXLFP_DIR}/lfpFunctor.h)
  target_compile_definitions(Fivox PUBLIC FIVOX_USE_LFP)
endif()

add_subdirectory(livre)
