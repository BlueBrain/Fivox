# Copyright (c) BBP/EPFL 2014-2016
#                        Jafet.VillafrancaDiaz@epfl.ch

include(CMakeDependentOption)

if(NOT FIVOX_USE_CUDA OR NOT CUDA_FOUND)
  message(STATUS "Disable cuda, used ${FIVOX_USE_CUDA}, found ${CUDA_FOUND}")
  return()
endif()
if(CUDA_VERSION_MAJOR VERSION_LESS 4)
  message(WARNING "CUDA 4.0 or later needed, CUDA support disabled")
  return()
endif()

add_definitions(-DUSE_CUDA)

cuda_include_directories(${PROJECT_SOURCE_DIR}/fivox/cuda)
cuda_include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

# Strangely, only relative paths work correctly for .cu files with
# cuda_add_library
set(CUDA_SOURCES simpleLFP.cu)

common_cuda_compile_options()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
cuda_add_library(FivoxCuda SHARED ${CUDA_SOURCES})
target_link_libraries(FivoxCuda Lunchbox)
list(APPEND FIVOX_LINK_LIBRARIES FivoxCuda)
set(FIVOX_LINK_LIBRARIES "${FIVOX_LINK_LIBRARIES}" PARENT_SCOPE)

install(TARGETS FivoxCuda EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION lib COMPONENT COMMON)

# Add a special target to clean nvcc generated files.
cuda_build_clean_target()
