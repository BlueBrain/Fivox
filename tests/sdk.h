
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "test.h"
#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <BBP/TestDatasets.h>
#include <itkImageFileWriter.h>
#include <boost/make_shared.hpp>

namespace
{
static const std::string targetName( "L5CSPC" );

template< typename T, typename K >
inline void _testSDKKernel( const size_t size )
{
    typedef itk::Image< T, 3 > Image;
    typedef fivox::EventFunctor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    fivox::EventSourcePtr source = boost::make_shared< K >(
                                 bbp::test::getBlueconfig(), targetName, 1.5f );
    filter->GetFunctor().setSource( source );

    // set up size and origin for loaded circuit
    const fivox::Vector3f& min = source->getMin();
    const fivox::Vector3f& bbox = source->getMax() - min;
    const float extent = bbox.find_max();
    const float position = min.find_min();
    BOOST_CHECK_GT( extent,  0.f );

    typename Image::SpacingType spacing;
    spacing.Fill( extent / float( size ));
    output->SetSpacing( spacing );

    typename Image::PointType origin;
    origin.Fill( position );
    output->SetOrigin( origin );

    std::ostringstream os;
    os << targetName << '_' << size << '_' << typeid( K ).name() << '_'
       << typeid( T ).name() << ".mhd";

    typedef itk::ImageFileWriter< Image > Writer;
    typename Writer::Pointer writer = Writer::New();
    writer->SetInput( filter->GetOutput( ));
    writer->SetFileName( os.str( ));

    writer->Update();
}
}
