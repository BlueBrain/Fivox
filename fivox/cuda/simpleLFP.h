
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

#define gpuErrchk( ans ) { gpuAssert(( ans ), __FILE__, __LINE__); }
inline void gpuAssert( cudaError_t code, const char *file, const int line )
{
    if( code != cudaSuccess )
    {
        std::cerr << "GPUassert: " << cudaGetErrorString( code ) << ", "
                  << file << ":" << line << std::endl;
        exit( code );
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

float simpleLFP( float* posX, float* posY, float* posZ,
                 float* radii, float* values, const Parameters& parameters,
                 const VolumeInfo& volInfo, float* output );
}
