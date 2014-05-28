// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_INPLACENEIGHBORHOODFILTER_H
#define FIVOX_INPLACENEIGHBORHOODFILTER_H

#include <itkInPlaceImageFilter.h>

namespace fivox
{
/** In-place image filter applying a neighborhood operator. */
template< typename TImage >
class InPlaceNeighborhoodFilter : public itk::InPlaceImageFilter< TImage,
                                                                  TImage >
{
public:
  /** Standard class typedefs. */
  typedef InPlaceNeighborhoodFilter                       Self;
  typedef itk::InPlaceImageFilter< TImage, TImage >       Superclass;
  typedef itk::SmartPointer< Self >                       Pointer;
  typedef itk::SmartPointer< const Self >                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(InPlaceNeighborhoodFilter, InPlaceImageFilter);

  /** Typedefs from Superclass */
  typedef typename Superclass::InputImagePointer  InputImagePointer;
  typedef typename Superclass::OutputImagePointer OutputImagePointer;

  /** Typedef to describe the output and input image region types. */
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
  itkStaticConstMacro(ImageDimension, unsigned int, ImageType::ImageDimension);

  /** Set/Get the destination index (where in the first input the second
   * input will be pasted. */
  itkSetMacro(DestinationIndex, ImageIndexType);
  itkGetConstMacro(DestinationIndex, ImageIndexType);

  /** Set/Get the source region (what part of the second input will be
   * pasted. */
  itkSetMacro(SourceRegion, ImageRegionType);
  itkGetConstMacro(SourceRegion, ImageRegionType);

protected:
  InPlaceNeighborhoodFilter();
  ~InPlaceNeighborhoodFilter() {}
  void PrintSelf(std::ostream & os, itk::Indent indent) const;

  /** InPlaceNeighborhoodFilter is implemented as a multithreaded filter. */
  void ThreadedGenerateData(const ImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId);

  ImageRegionType m_SourceRegion;
  ImageIndexType m_DestinationIndex;

private:
  InPlaceNeighborhoodFilter(const Self &); //purposely not implemented
  void operator=(const Self &);   //purposely not implemented
};
} // end namespace fivox

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "inPlaceNeighborhoodFilter.hxx"
#  endif
#endif
