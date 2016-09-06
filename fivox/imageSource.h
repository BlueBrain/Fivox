
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

#include <fivox/itk.h>
#include <fivox/types.h>
#include <fivox/progressObserver.h> // member
#include <lunchbox/monitor.h> // member

namespace fivox
{

/** ITK image source using an EventFunctor on each pixel to generate the output */
template< typename TImage >
class ImageSource : public itk::ImageSource< TImage >
{
public:
    /** Standard class typedefs. */
    typedef ImageSource                              Self;
    typedef itk::ImageSource< TImage >               Superclass;
    typedef itk::SmartPointer< Self >                Pointer;
    typedef itk::SmartPointer< const Self >          ConstPointer;
    typedef EventFunctor< TImage >                   Functor;
    typedef std::shared_ptr< Functor >               FunctorPtr;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(ImageSource, itk::ImageSource);

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

    /** @return the functor executed for each pixel during update. */
    FunctorPtr getFunctor();

    /** Set a new functor. */
    void setFunctor( FunctorPtr functor );

    /** Enable display of progress bar during voxelization. */
    void showProgress();

    /** Setup size and resolultion of output volume depending on user input. */
    void setup( const URIHandler& params );

    /** @return the bounding box of the data in micrometers. */
    const AABBf& getBoundingBox() const;

    /** @return the size of the output volume in micrometers. */
    const Vector3f& getSizeInMicrometer() const;

    /** @return the size of the output volume in voxels. */
    const Vector3ui& getSizeInVoxel() const;

    /** @return the resolution of the output volume in voxels per micrometer. */
    const Vector3f& getResolution() const;

    const itk::ImageRegionSplitterBase* GetImageRegionSplitter() const override
        { return _splitter; }

protected:
    ImageSource();
    virtual ~ImageSource() {}

    void PrintSelf(std::ostream & os, itk::Indent indent) const override;

    /** ImageSource is implemented as a multithreaded filter. */
    void ThreadedGenerateData( const ImageRegionType& outputRegionForThread,
                               itk::ThreadIdType threadId ) override;

    void BeforeThreadedGenerateData() override;

private:
    ImageSource(const Self &); //purposely not implemented
    void operator=(const Self &);   //purposely not implemented

    FunctorPtr _functor;
    itk::ImageRegionSplitterBase::Pointer _splitter;
    ProgressObserver::Pointer _progressObserver;
    lunchbox::Monitor< size_t > _completed;

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
