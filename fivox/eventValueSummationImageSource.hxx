
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Daniel.Nachbaur@epfl.ch
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

#ifndef FIVOX_EVENTVALUESUMMATIONIMAGESOURCE_HXX
#define FIVOX_EVENTVALUESUMMATIONIMAGESOURCE_HXX

#include "eventValueSummationImageSource.h"

#include <itkProgressReporter.h>

namespace fivox
{

template< typename TImage >
EventValueSummationImageSource< TImage >::EventValueSummationImageSource()
    : ImageSource< TImage >()
{
}

template< typename TImage >
void EventValueSummationImageSource< TImage >::GenerateData()
{
    Superclass::_progressObserver->reset();

    auto image = Superclass::GetOutput();
    image->Allocate();
    image->FillBuffer( 0 );

    auto source = Superclass::_eventSource;
    itk::ProgressReporter progress( this, 0, source->getNumChunks( ));
    size_t totalEvents = 0;
    typename TImage::PixelType maxValue = 0;
    for( size_t i = 0; i < source->getNumChunks(); ++i )
    {
        totalEvents += source->load( i, 1 );

        const float* __restrict__ posx = source->getPositionsX();
        const float* __restrict__ posy = source->getPositionsY();
        const float* __restrict__ posz = source->getPositionsZ();
        const float* __restrict__ values = source->getValues();

        #pragma vector aligned
        for( size_t j = 0; j < source->getNumEvents(); ++j )
        {
            typename TImage::PointType point;
            point[0] = posx[j];
            point[1] = posy[j];
            point[2] = posz[j];
            typename Superclass::ImageIndexType index;
            if( image->TransformPhysicalPointToIndex( point, index ))
            {
                const typename TImage::PixelType value =
                        image->GetPixel( index ) + values[j];
                maxValue = std::max( maxValue, value );
                image->SetPixel( index, value );
            }
        }

        progress.CompletedPixel();
    }

    LBINFO << "Voxelized " << totalEvents << " events for "
           << source->getNumChunks() << " chunks, max value " << maxValue
           << std::endl;
}

} // end namespace fivox

#endif
