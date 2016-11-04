
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

#include "simpleLFP.h"

namespace cuda
{
__global__ void kernelLFP( const float* __restrict__ eventsX,
                           const float* __restrict__ eventsY,
                           const float* __restrict__ eventsZ,
                           const float* __restrict__ radii,
                           const float* __restrict__ values,
                           const Parameters& params, const VolumeInfo& volInfo,
                           float* __restrict__ output )
{
    // 1D grid of 1D blocks
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numVoxels = volInfo.dimensions.x *
                                   volInfo.dimensions.y *
                                   volInfo.dimensions.z;
    if( threadId >= numVoxels )
        return;

    const int xIndex = threadId % volInfo.dimensions.x;
    const int yIndex = ( threadId / volInfo.dimensions.x )
                       % volInfo.dimensions.y;
    const int zIndex = (( threadId / volInfo.dimensions.x )
                        / volInfo.dimensions.y ) % volInfo.dimensions.z;

    const float voxelPosX = xIndex * volInfo.voxelSize + volInfo.origin.x;
    const float voxelPosY = yIndex * volInfo.voxelSize + volInfo.origin.y;
    const float voxelPosZ = zIndex * volInfo.voxelSize + volInfo.origin.z;

    // Compute directly the inverted value to gain performance in the for loop
    const float cutOffDistance = __frcp_rn( params.cutoff );

    float current( 0.f );
    for( unsigned int i = 0; i < params.numEvents; ++i )
    {
        const float value( values[i] );

        const float distanceX = voxelPosX - eventsX[i];
        const float distanceY = voxelPosY - eventsY[i];
        const float distanceZ = voxelPosZ - eventsZ[i];

        const float distance2( distanceX * distanceX +
                               distanceY * distanceY +
                               distanceZ * distanceZ );

        // Use the reciprocal of sqrt
        const float length = __frsqrt_rn( distance2 );
        // Comparison is inverted, as we are using the reciprocal values
        if( length < cutOffDistance )
            continue;

        // If center of the voxel within the event radius, use the
        // voltage at the surface of the compartment (at 'radius' distance)
        const float radius( radii[i] );
        // Comparison is inverted, as we are using the reciprocal values
        // (radius is already stored like that from the loader)
        current += value * min( radius, length ); // mA
    }
    // voltageFactor =  1 / (4 * PI * conductivity),
    // with conductivity = 1 / 3.54 (siemens per meter)
    const float voltageFactor = 0.281704249f;
    // output voltage in mV
    output[threadId] = voltageFactor * current;
}

float simpleLFP( float* posX, float* posY, float* posZ, float* radii,
                 float* values, const Parameters& parameters,
                 const VolumeInfo& volInfo, float* output )
{
    cuda::Parameters* cudaParameters;
    cuda::VolumeInfo* cudaVolInfo;
    gpuErrchk( cudaMalloc( (void**)&cudaParameters, sizeof( cuda::Parameters)));
    gpuErrchk( cudaMalloc( (void**)&cudaVolInfo, sizeof( cuda::VolumeInfo )));

    gpuErrchk( cudaMemcpy( cudaParameters, &parameters,
                           sizeof( cuda::Parameters ), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy( cudaVolInfo, &volInfo, sizeof( cuda::VolumeInfo ),
                           cudaMemcpyHostToDevice ));

    const int numVoxels = volInfo.dimensions.x *
                          volInfo.dimensions.y *
                          volInfo.dimensions.z;

    // 1D grid of 1D blocks
    dim3 blockSize( 512, 1 );
    dim3 gridSize( ( numVoxels + blockSize.x * blockSize.y - 1 ) /
                   ( blockSize.x * blockSize.y ), 1 );

    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate( &start ));
    gpuErrchk( cudaEventCreate( &stop ));

    gpuErrchk( cudaEventRecord( start ));
    kernelLFP<<<gridSize, blockSize>>>( posX, posY, posZ, radii, values,
                                        *cudaParameters, *cudaVolInfo, output );
    gpuErrchk( cudaPeekAtLastError( ));
    gpuErrchk( cudaEventRecord( stop ));
    gpuErrchk( cudaEventSynchronize( stop ));

    float milliseconds = 0;
    gpuErrchk( cudaEventElapsedTime( &milliseconds, start, stop ));

    gpuErrchk( cudaEventDestroy( start ));
    gpuErrchk( cudaEventDestroy( stop ));

    gpuErrchk( cudaFree( cudaParameters ));
    gpuErrchk( cudaFree( cudaVolInfo ));

    return milliseconds;
}

}
