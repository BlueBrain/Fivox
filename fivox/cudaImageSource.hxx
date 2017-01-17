
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

#ifndef FIVOX_CUDAIMAGESOURCE_HXX
#define FIVOX_CUDAIMAGESOURCE_HXX

#include "cuda/simpleLFP.h"
#include "cudaImageSource.h"

namespace fivox
{
template< typename TImage >
CudaImageSource< TImage >::CudaImageSource()
    : ImageSource< TImage >()
{
}

template< typename TImage >
void CudaImageSource< TImage >::GenerateData()
{
    auto image = Superclass::GetOutput();
    image->Allocate();
    image->FillBuffer( 0 );

    typename TImage::SizeType vSize = image->GetRequestedRegion().GetSize();
    const size_t width = vSize[0];
    const size_t height = vSize[1];
    const size_t depth = vSize[2];

    cuda::VolumeInfo volInfo;
    volInfo.dimensions.x = width;
    volInfo.dimensions.y = height;
    volInfo.dimensions.z = depth;

    volInfo.voxelSize = image->GetSpacing()[0];

    const auto& origin = image->GetOrigin();
    volInfo.origin.x = origin[0];
    volInfo.origin.y = origin[1];
    volInfo.origin.z = origin[2];

    auto source = Superclass::_eventSource;
    source->load();
    const int fsize = source->getNumEvents() * sizeof(float);

    cuda::Parameters parameters;
    parameters.numEvents = source->getNumEvents();
    parameters.cutoff = source->getCutOffDistance();

    float* posX;
    float* posY;
    float* posZ;
    float* radii;
    float* values;
    // copy input from host to device
    gpuErrchk( cudaMalloc( (void**)&posX, fsize ));
    gpuErrchk( cudaMalloc( (void**)&posY, fsize ));
    gpuErrchk( cudaMalloc( (void**)&posZ, fsize ));
    gpuErrchk( cudaMalloc( (void**)&radii, fsize ));
    gpuErrchk( cudaMalloc( (void**)&values, fsize ));
    gpuErrchk( cudaMemcpy( posX, source->getPositionsX(), fsize,
                           cudaMemcpyHostToDevice ));
    gpuErrchk( cudaMemcpy( posY, source->getPositionsY(), fsize,
                           cudaMemcpyHostToDevice ));
    gpuErrchk( cudaMemcpy( posZ, source->getPositionsZ(), fsize,
                           cudaMemcpyHostToDevice ));
    gpuErrchk( cudaMemcpy( radii, source->getRadii(), fsize,
                           cudaMemcpyHostToDevice ));
    gpuErrchk( cudaMemcpy( values, source->getValues(), fsize,
                           cudaMemcpyHostToDevice ));

    const int numVoxels = width * height * depth;

    float* cudaOutput;
    gpuErrchk( cudaMalloc( (void**)&cudaOutput, numVoxels * sizeof(float) ));
    const float elapsed = cuda::simpleLFP( posX, posY, posZ, radii, values,
                                           parameters, volInfo, cudaOutput );
    LBINFO << "CUDA elapsed time: " << elapsed << "ms" << std::endl;

    // copy output from device to host
    float* output = (float*)malloc( numVoxels * sizeof(float) );
    gpuErrchk( cudaMemcpy( output, cudaOutput, numVoxels * sizeof(float),
                           cudaMemcpyDeviceToHost ));

    gpuErrchk( cudaFree( posX ));
    gpuErrchk( cudaFree( posY ));
    gpuErrchk( cudaFree( posZ ));
    gpuErrchk( cudaFree( radii ));
    gpuErrchk( cudaFree( values ));
    gpuErrchk( cudaFree( cudaOutput ));

    for( size_t i = 0; i < width; ++i )
        for( size_t j = 0; j < height; ++j )
            for( size_t k = 0; k < depth; ++k )
            {
                typename Superclass::ImageIndexType index;
                index[0] = i;
                index[1] = j;
                index[2] = k;
                const size_t flatIndex = i + j * width + k * width * height;
                image->SetPixel( index, output[flatIndex] );
            }
    free( output );

    LBINFO << "Voxelized " << source->getNumEvents() << " events" << std::endl;
}

} // end namespace fivox

#endif
