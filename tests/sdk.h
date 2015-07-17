
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "test.h"
#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <itkImageFileWriter.h>
#include <boost/make_shared.hpp>

namespace
{

template< typename T >
inline void _testSDKKernel( const size_t size, fivox::EventSourcePtr source,
                            const float expectedValue )
{
    typedef itk::Image< T, 3 > Image;
    typedef fivox::EventFunctor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );
    filter->GetFunctor().setSource( source );

    // set up size and origin for loaded circuit
    const fivox::AABBf& bbox = source->getBoundingBox();
    const fivox::Vector3f& position = bbox.getMin();
    const float extent = bbox.getDimension().find_max();
    BOOST_CHECK_GT( extent,  0.f );

    typename Image::SpacingType spacing;
    spacing.Fill( extent / float( size ));
    output->SetSpacing( spacing );

    typename Image::PointType origin;
    origin[0] = position[0];
    origin[1] = position[1];
    origin[2] = position[2];
    output->SetOrigin( origin );

#ifdef NDEBUG
    filter->Update();
#else
    std::ostringstream os;
    os << size << '_' << typeid( *source ).name() << '_' << typeid( T ).name()
       << ".mhd";

    typedef itk::ImageFileWriter< Image > Writer;
    typename Writer::Pointer writer = Writer::New();
    writer->SetInput( filter->GetOutput( ));
    writer->SetFileName( os.str( ));

    writer->Update();
#endif
    if( size == 8 )
    {
        const typename Image::IndexType index = {{ 2, 2, 3 }};
        const float value = float( output->GetPixel( index ));
        BOOST_CHECK_CLOSE( value, expectedValue, 1/*%*/ );
    }
}
}
