
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_IMAGESOURCE_H
#define FIVOX_IMAGESOURCE_H

#include <fivox/itk.h>
#include <fivox/types.h>

namespace fivox
{
/** Image source using an EventFunctor on each pixel to generate the output */
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

    virtual const itk::ImageRegionSplitterBase* GetImageRegionSplitter() const
        { return _splitter; }

protected:
    ImageSource();
    ~ImageSource() {}

    void PrintSelf(std::ostream & os, itk::Indent indent) const;

    /** ImageSource is implemented as a multithreaded filter. */
    void ThreadedGenerateData( const ImageRegionType& outputRegionForThread,
                               itk::ThreadIdType threadId );

private:
    ImageSource(const Self &); //purposely not implemented
    void operator=(const Self &);   //purposely not implemented

    FunctorPtr _functor;
    itk::ImageRegionSplitterBase::Pointer _splitter;
};
} // end namespace fivox

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "imageSource.hxx"
#  endif
#endif
