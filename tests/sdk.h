
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */

#include "test.h"
#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <itkImageFileWriter.h>
#include <itkStatisticsImageFilter.h>

namespace
{

template< typename T >
inline void _testSDKKernel( const size_t size, fivox::EventSourcePtr source,
                            const float expectedValue LB_UNUSED )
{
    typedef itk::Image< T, 3 > Image;
    typedef fivox::ImageSource< Image > Filter;
    typedef typename Filter::Functor Functor;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );
    filter->getFunctor()->setSource( source );

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
    if( size == 8 )
    {
        typedef itk::StatisticsImageFilter< Image > StatisticsFilterType;
        typename StatisticsFilterType::Pointer stat =
            StatisticsFilterType::New();
        stat->SetInput( output );
        stat->Update();

        BOOST_CHECK_CLOSE( stat->GetMean(), expectedValue, 1/*%*/ );
    }
#endif
}
}
