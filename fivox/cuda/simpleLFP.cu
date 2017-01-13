
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

    const unsigned int activeThreads = min(blockDim.x,
                                           numVoxels - blockIdx.x * blockDim.x);
    const unsigned int nPasses =
        ( params.numEvents + activeThreads - 1 )  / activeThreads;

    float current( 0.f );
    const unsigned int sharedEventIndex = threadIdx.x * 5;
    extern __shared__ float sharedEvents[];
    for( unsigned int i = 0; i < nPasses; ++i )
    {
        const unsigned int eventIndex = i * activeThreads + threadIdx.x;

        if( eventIndex < params.numEvents )
        {
            sharedEvents[ sharedEventIndex ] =  eventsX[ eventIndex ];
            sharedEvents[ sharedEventIndex + 1 ] =  eventsY[ eventIndex ];
            sharedEvents[ sharedEventIndex + 2 ] =  eventsZ[ eventIndex ];
            sharedEvents[ sharedEventIndex + 3 ] =  radii[ eventIndex ];
            sharedEvents[ sharedEventIndex + 4 ] =  values[ eventIndex ];
        }
        __syncthreads();

        if( eventIndex >= params.numEvents )
            break;

        for( unsigned int j = 0; j < activeThreads; ++j )
        {
            const unsigned int index = j * 5;
            const float value( sharedEvents[ index + 4 ]);

            const float distanceX = voxelPosX - sharedEvents[ index ];
            const float distanceY = voxelPosY - sharedEvents[ index + 1 ];
            const float distanceZ = voxelPosZ - sharedEvents[ index + 2 ];

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
            const float radius( sharedEvents[ index + 3 ] );
            // Comparison is inverted, as we are using the reciprocal values
            // (radius is already stored like that from the loader)
            current += value * min( radius, length ); // mA
        }
        __syncthreads();
    }
    // voltageFactor =  1 / (4 * PI * conductivity),
    // with conductivity = 1 / 3.54 (siemens per meter)
    const float voltageFactor = 0.281704249f;
    // output voltage in mV
    output[threadId] = voltageFactor * current;
}

float simpleLFP( const float* posX, const float* posY, const float* posZ,
                 const float* radii, const float* values,
                 const Parameters& parameters, const VolumeInfo& volInfo,
                 float* output )
{
    cuda::Parameters* cudaParameters;
    cuda::VolumeInfo* cudaVolInfo;
    gpuErrchk( cudaMalloc( (void**)&cudaParameters, sizeof( cuda::Parameters )));
    gpuErrchk( cudaMalloc( (void**)&cudaVolInfo, sizeof( cuda::VolumeInfo )));

    gpuErrchk( cudaMemcpy( cudaParameters, &parameters,
                           sizeof( cuda::Parameters ), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy( cudaVolInfo, &volInfo, sizeof( cuda::VolumeInfo ),
                           cudaMemcpyHostToDevice ));

    const int numVoxels = volInfo.dimensions.x *
                          volInfo.dimensions.y *
                          volInfo.dimensions.z;

    // 1D grid of 1D blocks
    unsigned int nThreads = 512;
    unsigned int gridSize = ( numVoxels + nThreads - 1 ) / nThreads;

    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate( &start ));
    gpuErrchk( cudaEventCreate( &stop ));

    gpuErrchk( cudaEventRecord( start ));
    kernelLFP<<< gridSize, nThreads, nThreads * 5 * sizeof(float) >>>
                                    ( posX, posY, posZ, radii, values,
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
