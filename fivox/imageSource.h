
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

#ifndef FIVOX_IMAGESOURCE_H
#define FIVOX_IMAGESOURCE_H

#include <fivox/types.h>
#include <fivox/progressObserver.h> // member

#include <itkImageSource.h> // base class

namespace fivox
{

/** Base class for any image source to sample data from an event source. */
template< typename TImage >
class ImageSource : public itk::ImageSource< TImage >
{
public:
    /** Standard class typedefs. */
    typedef ImageSource                              Self;
    typedef itk::ImageSource< TImage >               Superclass;
    typedef itk::SmartPointer< Self >                Pointer;
    typedef itk::SmartPointer< const Self >          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ImageSource, itk::ImageSource)

    /** Typedef to describe the output image region types. */
    typedef TImage                          ImageType;
    typedef typename ImageType::RegionType  ImageRegionType;

    typedef typename ImageType::Pointer      ImagePointer;
    typedef typename ImageType::ConstPointer ImageConstPointer;

    /** Typedef to describe the type of pixel. */
    typedef typename ImageType::PixelType ImagePixelType;

    /** Typedef to describe the output and input image index and size types. */
    typedef typename ImageType::IndexType ImageIndexType;
    typedef typename ImageType::SizeType  ImageSizeType;

    /** ImageDimension enumeration */
    itkStaticConstMacro( ImageDimension, unsigned int,
                         ImageType::ImageDimension );

    /** Set the event source that is used for sampling into the volume. */
    void setEventSource( EventSourcePtr source ) { _eventSource = source; }

    /** @return the event source used for sampling. */
    EventSourcePtr getEventSource() { return _eventSource; }

    /**
     * Setup size and resolution of output volume depending on user input.
     *
     * setEventSource() must be called before.
     */
    void setup( const URIHandler& params );

    /** @return the bounding box of the data in micrometers. */
    const AABBf& getBoundingBox() const;

    /** @return the size of the output volume in micrometers. */
    const Vector3f& getSizeInMicrometer() const;

    /** @return the size of the output volume in voxels. */
    const Vector3ui& getSizeInVoxel() const;

    /** @return the resolution of the output volume in voxels per micrometer. */
    const Vector3f& getResolution() const;

protected:
    ImageSource();
    ImageSource( const Self& ) = delete;
    void operator=( const Self& ) = delete;

    void PrintSelf( std::ostream & os, itk::Indent indent ) const override;

    EventSourcePtr _eventSource;
    ProgressObserver::Pointer _progressObserver;

    AABBf _boundingBox;
    Vector3ui _sizeVoxel;
    Vector3f _sizeMicrometer;
    Vector3f _resolution;
};
} // end namespace fivox

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "imageSource.hxx"
#  endif
#endif
