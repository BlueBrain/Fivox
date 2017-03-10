
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, const int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << ", " << file
                  << ":" << line << std::endl;
        exit(code);
    }
}

namespace cuda
{
struct Parameters
{
    size_t numEvents;
    float cutoff;
};

struct VolumeInfo
{
    uint3 dimensions;
    float voxelSize;
    float3 origin;
};

/**
 * Compute the local field potential from a set of input events
 *
 * @param posX array of x-coordinates of event positions
 * @param posY array of y-coordinates of event positions
 * @param posZ array of z-coordinates of event positions
 * @param radii array of event radii
 * @param values array of event values
 * @param parameters structure containing the number of events and the cutoff
 * distance
 * @param volInfo structure containing the volume dimensions, the voxel size and
 * the origin coordinates (bottom left corner)
 * @param output array of the output values. Each value corresponds to a voxel,
 * being its 3D indices flattened: first traversing along X, then Y
 * (using "width" steps), and finally Z (using "width"*"height" planes)
 * @return the time of execution, in milliseconds, of the computation in the
 * CUDA kernel
 */
float simpleLFP(const float* posX, const float* posY, const float* posZ,
                const float* radii, const float* values,
                const Parameters& parameters, const VolumeInfo& volInfo,
                float* output);
}
