
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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

#ifndef FIVOX_FUNCTORIMAGESOURCE_HXX
#define FIVOX_FUNCTORIMAGESOURCE_HXX

#include "functorImageSource.h"

#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkProgressReporter.h>

namespace fivox
{
static const int _splitDirection = 2; // fastest in latest test

template< typename TImage > FunctorImageSource< TImage >::FunctorImageSource()
    : ImageSource< TImage >()
{
    itk::ImageRegionSplitterDirection::Pointer splitter =
        itk::ImageRegionSplitterDirection::New();
    splitter->SetDirection( _splitDirection );
    _splitter = splitter;
}

template< typename TImage > typename FunctorImageSource< TImage >::FunctorPtr
FunctorImageSource< TImage >::getFunctor()
{
    return _functor;
}

template< typename TImage >
void FunctorImageSource< TImage >::setFunctor( FunctorPtr functor )
{
    _functor = functor;
}

template< typename TImage >
void FunctorImageSource< TImage >::ThreadedGenerateData(
    const typename Superclass::ImageRegionType& outputRegionForThread,
    const itk::ThreadIdType threadId )
{
    typename Superclass::ImagePointer image = Superclass::GetOutput();
    typedef itk::ImageLinearIteratorWithIndex< TImage > ImageIterator;
    ImageIterator i( image, outputRegionForThread );
    i.SetDirection(0);
    i.GoToBegin();

    const size_t nLines = image->GetRequestedRegion().GetSize()[1] *
                          image->GetRequestedRegion().GetSize()[2];
    itk::ProgressReporter progress( this, threadId, nLines );
    size_t totalLines = 0;

    while( !i.IsAtEnd( ))
    {
        const typename Superclass::ImageIndexType& index = i.GetIndex();

        const typename TImage::SpacingType spacing = image->GetSpacing();
        typename TImage::PointType point;
        image->TransformIndexToPhysicalPoint( index, point );

        i.Set( (*_functor)( point, spacing ) );

        ++i;
        if( i.IsAtEndOfLine( ))
        {
            i.NextLine();
            // report progress only once per line for lower contention on
            // monitor. Main thread reports to itk, all others to the monitor.
            if( threadId == 0 )
            {
                size_t done = _completed.set( 0 ) + 1 /*self*/;
                totalLines += done;
                while( done-- )
                    progress.CompletedPixel();
            }
            else
                ++_completed;
        }
    }

    if( threadId == 0 )
    {
        while( totalLines < nLines )
        {
            _completed.waitNE( 0 );
            size_t done = _completed.set( 0 );
            totalLines += done;
            while( done-- )
                progress.CompletedPixel();
        }
    }
}

template< typename TImage >
void FunctorImageSource< TImage >::BeforeThreadedGenerateData()
{
    // load all the data of the current frame
    auto source = Superclass::_eventSource;
    if( !source )
        return;

    const ssize_t updatedEvents = source->load();
    const float time = source->getCurrentTime();
    if( updatedEvents < 0 )
    {
        LBERROR << "Timestamp " << time << "ms not loaded, no data or events"
                << std::endl;
    }
    else
    {
        LBINFO << "Timestamp " << time << "ms loaded, updated " << updatedEvents
               << " event(s)" << std::endl;
    }

    _completed = 0;
    _functor->beforeGenerate();
    Superclass::_progressObserver->reset();
}

} // end namespace fivox

#endif
