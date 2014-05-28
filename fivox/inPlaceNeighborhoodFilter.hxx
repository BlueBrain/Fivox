// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_INPLACENEIGHBORHOODFILTER_HXX
#define FIVOX_INPLACENEIGHBORHOODFILTER_HXX


#include "inPlaceNeighborhoodFilter.h"
// #include "itkObjectFactory.h"
#include "itkProgressReporter.h"
// #include "itkImageAlgorithm.h"

namespace fivox
{
template< typename TImage >
InPlaceNeighborhoodFilter< TImage >::InPlaceNeighborhoodFilter()
{
  itk::ProcessObject::SetNumberOfRequiredInputs( 1 );
  Superclass::InPlaceOn();
}

template< typename TImage >
void InPlaceNeighborhoodFilter< TImage >::PrintSelf(std::ostream & os,
                                                    itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template< typename TImage >
void InPlaceNeighborhoodFilter< TImage >::ThreadedGenerateData(
  const ImageRegionType& /*outputRegionForThread*/, itk::ThreadIdType threadId)
{
  itkDebugMacro(<< "Actually executing");

  // support progress methods/callbacks
  itk::ProgressReporter progress( this, threadId, 1 );

  progress.CompletedPixel();
}
} // end namespace fivox

#endif
