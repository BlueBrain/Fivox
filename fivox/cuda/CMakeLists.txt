# Copyright (c) BBP/EPFL 2014-2016
#                        Jafet.VillafrancaDiaz@epfl.ch

cuda_include_directories(${PROJECT_SOURCE_DIR}/fivox/cuda)

# Strangely, only relative paths work correctly for .cu files with
# cuda_add_library
set(CUDA_SOURCES simpleLFP.cu)

common_cuda_compile_options()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES
                                        -gencode arch=compute_20,code=sm_20
                                        -gencode arch=compute_30,code=sm_30
                                        -gencode arch=compute_35,code=sm_35
                                        -gencode arch=compute_50,code=sm_50")
cuda_add_library(FivoxCuda SHARED ${CUDA_SOURCES})

install(TARGETS FivoxCuda EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION lib COMPONENT COMMON)

# Add a special target to clean nvcc generated files.
if(NOT TARGET clean_cuda_depends)
  cuda_build_clean_target()
endif()
