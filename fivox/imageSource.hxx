// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_IMAGESOURCE_HXX
#define FIVOX_IMAGESOURCE_HXX


#include "imageSource.h"

#include <itkProgressReporter.h>
#include <itkImageLinearIteratorWithIndex.h>

namespace fivox
{
template< typename TImage, typename TFunctor >
ImageSource< TImage, TFunctor >::ImageSource()
{
  // set up default size
  static const size_t size = 128;
  typename TImage::SizeType vSize;
  vSize.Fill( size );
  typename TImage::RegionType region;
  region.SetSize( vSize );

  typename TImage::Pointer output = Superclass::GetOutput();
  output->SetRequestedRegion( region );
}

template< typename TImage, typename TFunctor >
void ImageSource< TImage, TFunctor >::PrintSelf(std::ostream & os,
                                                itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template< typename TImage, typename TFunctor >
void ImageSource< TImage, TFunctor >::ThreadedGenerateData(
  const ImageRegionType& outputRegionForThread, itk::ThreadIdType threadId)
{
  itkDebugMacro(<< "Actually executing");

  ImagePointer image = Superclass::GetOutput();
  typedef itk::ImageLinearIteratorWithIndex< TImage > ImageIterator;
  ImageIterator i( image, outputRegionForThread );
  i.SetDirection(0);
  i.GoToBegin();

  itk::ProgressReporter progress( this, threadId,
                                  outputRegionForThread.GetNumberOfPixels( ));
  while( !i.IsAtEnd( ))
  {
    const ImageIndexType& index = i.GetIndex();

    typedef typename TImage::PointType PointType;
    PointType point;
    image->TransformIndexToPhysicalPoint( index, point );

    i.Set( m_Functor( point ));

    ++i;
    if( i.IsAtEndOfLine( ))
      i.NextLine();
    progress.CompletedPixel();
  }
}
} // end namespace fivox

#endif
