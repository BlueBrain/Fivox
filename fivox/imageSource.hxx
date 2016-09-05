
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
#include "uriHandler.h"

#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageFileReader.h>

#include <lunchbox/log.h>

namespace fivox
{
template< typename TImage > ImageSource< TImage >::ImageSource()
    : _progressObserver( ProgressObserver::New( ))
{
    // set up default size
    static const size_t size = 256;
    typename TImage::SizeType vSize;
    vSize.Fill( size );
    typename TImage::RegionType region;
    region.SetSize( vSize );

    typename TImage::Pointer output = Superclass::GetOutput();
    output->SetRequestedRegion( region );

    Superclass::AddObserver( itk::ProgressEvent(), _progressObserver );
}

template< typename TImage >
void ImageSource< TImage >::PrintSelf(std::ostream & os, itk::Indent indent )
    const
{
    Superclass::PrintSelf( os, indent );
}

template< typename TImage >
void ImageSource< TImage >::setup( const URIHandler& params )
{
    _progressObserver->enablePrint();

    const std::string& refVolume = params.getReferenceVolume();
    if( refVolume.empty( ))
    {
        _boundingBox = _eventSource->getBoundingBox();
        _sizeMicrometer = _boundingBox.getSize() +
                          params.getExtendDistance() * 2.f;

        if( params.getSizeInVoxel() > 0 )
        {
            for( size_t i = 0; i < 3; ++i )
                _sizeVoxel[i] = params.getSizeInVoxel() *
                                _sizeMicrometer[i] / _sizeMicrometer.find_max();
            const size_t maxSizeIndex = _sizeMicrometer.find_max_index();
            _sizeVoxel[maxSizeIndex] = params.getSizeInVoxel();
        }
        else
            _sizeVoxel = _sizeMicrometer * params.getResolution();
        _resolution = Vector3f(_sizeVoxel) / _sizeMicrometer;

        return;
    }

    LBINFO << "Using reference volume '" << refVolume
           << "' to setup volume size and resolution" << std::endl;

    typedef itk::Image< double, 3 > ImageType;
    typedef itk::ImageFileReader< ImageType > ReaderType;

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( refVolume );
    reader->Update();
    const auto& referenceVolume = reader->GetOutput();

    const auto& region = referenceVolume->GetLargestPossibleRegion();
    const auto& size = region.GetSize();
    const auto& spacing = referenceVolume->GetSpacing();

    _sizeMicrometer = Vector3f( size[0] * spacing[0],
                                size[1] * spacing[1],
                                size[2] * spacing[2] );
    _boundingBox = AABBf( Vector3f(0.f), _sizeMicrometer );
    _sizeVoxel = Vector3ui( size[0], size[1], size[2] );
    _resolution = Vector3f( 1.f/spacing[0], 1.f/spacing[1], 1.f/spacing[2]);
}

template< typename TImage >
const AABBf& ImageSource< TImage >::getBoundingBox() const
{
    return _boundingBox;
}

template< typename TImage >
const Vector3ui& ImageSource< TImage >::getSizeInVoxel() const
{
    return _sizeVoxel;
}

template< typename TImage >
const Vector3f& ImageSource< TImage >::getSizeInMicrometer() const
{
    return _sizeMicrometer;
}

template< typename TImage >
const Vector3f& ImageSource< TImage >::getResolution() const
{
    return _resolution;
}

} // end namespace fivox

#endif
