
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

#ifndef FIVOX_IMAGESOURCE_HXX
#define FIVOX_IMAGESOURCE_HXX

#include "imageSource.h"

#include <fivox/densityFunctor.h>
#include <itkProgressReporter.h>
#include <itkImageLinearIteratorWithIndex.h>

namespace fivox
{
static const int _splitDirection = 2; // fastest in latest test

template< typename TImage > ImageSource< TImage >::ImageSource()
    : _functor( new DensityFunctor< TImage >( fivox::Vector2f( )))
{
    itk::ImageRegionSplitterDirection::Pointer splitter =
        itk::ImageRegionSplitterDirection::New();
    splitter->SetDirection( _splitDirection );
    _splitter = splitter;

    // set up default size
    static const size_t size = 256;
    typename TImage::SizeType vSize;
    vSize.Fill( size );
    typename TImage::RegionType region;
    region.SetSize( vSize );

    typename TImage::Pointer output = Superclass::GetOutput();
    output->SetRequestedRegion( region );
}

template< typename TImage >
typename ImageSource< TImage >::FunctorPtr ImageSource< TImage >::getFunctor()
{
    return _functor;
}

template< typename TImage >
void ImageSource< TImage >::setFunctor( FunctorPtr functor )
{
    _functor = functor;
}

template< typename TImage >
void ImageSource< TImage >::showProgress()
{
    _progressObserver = ProgressObserver::New();
    Superclass::AddObserver( itk::ProgressEvent(), _progressObserver );
}

template< typename TImage >
void ImageSource< TImage >::PrintSelf(std::ostream & os, itk::Indent indent )
    const
{
    Superclass::PrintSelf( os, indent );
}

template< typename TImage >
void ImageSource< TImage >::ThreadedGenerateData(
    const ImageRegionType& outputRegionForThread, itk::ThreadIdType threadId )
{
    ImagePointer image = Superclass::GetOutput();
    typedef itk::ImageLinearIteratorWithIndex< TImage > ImageIterator;
    ImageIterator i( image, outputRegionForThread );
    i.SetDirection(0);
    i.GoToBegin();

    const size_t nLines = this->GetOutput()->GetRequestedRegion().GetSize()[1] *
                          this->GetOutput()->GetRequestedRegion().GetSize()[2];
    itk::ProgressReporter progress( this, threadId, nLines );
    size_t totalLines = 0;

    while( !i.IsAtEnd( ))
    {
        const ImageIndexType& index = i.GetIndex();

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
void ImageSource< TImage >::BeforeThreadedGenerateData()
{
    _completed = 0;
    _functor->beforeGenerate();
    if( _progressObserver )
        static_cast< ProgressObserver& >(*_progressObserver).reset();
}

} // end namespace fivox

#endif
