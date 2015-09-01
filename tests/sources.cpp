
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */

#define BOOST_TEST_MODULE Sources

#include "test.h"
#include <fivox/compartmentLoader.h>
#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <fivox/somaLoader.h>
#include <fivox/spikeLoader.h>
#include <fivox/synapseLoader.h>
#include <BBP/TestDatasets.h>
#include <itkImageFileWriter.h>
#include <itkStatisticsImageFilter.h>
#include <itkTimeProbe.h>
#include <iomanip>

namespace
{
static const size_t _minResolution = 8;

template< typename T >
inline float _testSDKKernel(
    itk::SmartPointer< fivox::ImageSource< itk::Image< T, 3 >>> filter,
    const size_t size, const float expectedValue )
{
    typedef itk::Image< T, 3 > Image;
    typedef fivox::ImageSource< Image > Filter;
    typedef typename Filter::Functor Functor;

    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    // set up size and origin for loaded data
    fivox::EventSourcePtr source = filter->getFunctor()->getSource();
    source->load( 95.f );
    const fivox::AABBf& bbox = source->getBoundingBox();
    const fivox::Vector3f& position = bbox.getMin();
    const float extent = bbox.getDimension().find_max();
    BOOST_CHECK_GT( extent, 0.f );

    typename Image::SpacingType spacing;
    spacing.Fill( extent / float( size ));
    output->SetSpacing( spacing );

    typename Image::PointType origin;
    origin[0] = position[0];
    origin[1] = position[1];
    origin[2] = position[2];
    output->SetOrigin( origin );
    filter->Modified();

    itk::TimeProbe clock;
    clock.Start();
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
    clock.Stop();

    if( size == _minResolution )
    {
        typedef itk::StatisticsImageFilter< Image > StatisticsFilterType;
        typename StatisticsFilterType::Pointer stat =
            StatisticsFilterType::New();
        stat->SetInput( output );
        stat->Update();

        BOOST_CHECK_CLOSE( stat->GetMean(), expectedValue, 1/*%*/ );
    }

    return clock.GetTotal();
}
}

BOOST_AUTO_TEST_CASE( Sources )
{
    char** argv = boost::unit_test::framework::master_test_suite().argv;
    const bool unitTest = std::string( argv[0] ).find( "perf-" ) ==
                          std::string::npos;
    static const size_t maxSize = unitTest ? _minResolution : 1024;
    static const size_t maxThreads = unitTest ? 0 : 16;
    // Better, but not always available data:
    // "fivoxSynapses:///gpfs/bbp.cscs.ch/home/nachbaur/BlueConfig_3m,target=L23_DBC",
    // "fivoxSpikes:///gpfs/bbp.cscs.ch/home/nachbaur/BlueConfig_3m",
    const lunchbox::Strings uris = { "fivox://", "fivoxSomas://",
                                     "fivoxSpikes://", "fivoxSynapses://" };

    // Reference mean expected values for uris:
    const float byteRef[] = { 0.896484375f,
                              0.521484375f,
                              0.013671875f,
                              0.03515625f };
    const float floatRef[] = { 0.0047158297430449636f,
                               0.0034715187880252074f,
                               8.9044979517893807e-05f,
                               0.00017834029219887526f };

    std::cout.setf( std::ios::right, std::ios::adjustfield );
    std::cout.precision( 5 );
    std::cout << "    Test, byte MVox/sec, float MVox/sec" << std::endl;

    for( lunchbox::StringsCIter it = uris.begin(); it != uris.end(); ++it )
    {
        const size_t i = std::distance( uris.begin(), it );
        const fivox::URIHandler params( *it );
        auto filter1 = params.newImageSource< uint8_t >();
        auto filter2 = params.newImageSource< float >();

        std::cout << "   Size, " << *it << "," << std::endl;
        for( size_t j = _minResolution; j <= maxSize; j = j << 1 )
        {
            const float t1 = _testSDKKernel< uint8_t >( filter1, j,
                                                        byteRef[ i ] );
            const float t2 = _testSDKKernel< float >( filter2, j,
                                                      floatRef[ i ] );
            std::cout << std::setw( 8 ) << j << ',' << std::setw(14)
                      << j*j*j / 1024.f / 1024.f / t1 << ',' << std::setw(15)
                      << j*j*j / 1024.f / 1024.f / t2 << std::endl;
        }

        if( maxThreads )
            std::cout << "Threads, " << *it << "," << std::endl;
        const size_t size = maxSize >> 2;
        for( size_t j = 1; j <= maxThreads; j = j << 1 )
        {
            filter1->SetNumberOfThreads( maxThreads * j );
            filter2->SetNumberOfThreads( maxThreads * j );

            float t1 = _testSDKKernel< uint8_t >( filter1, size, byteRef[i]);
            float t2 = _testSDKKernel< float >( filter2, size, floatRef[i] );
            std::cout << std::setw(7) << filter1->GetNumberOfThreads() << ','
                      << std::setw(14) << size*size*size / 1024.f / 1024.f / t1
                      << ',' << std::setw(15)
                      << size*size*size / 1024.f / 1024.f / t2 << std::endl;

            filter1->SetNumberOfThreads( maxThreads / j );
            filter2->SetNumberOfThreads( maxThreads / j );
            t1 = _testSDKKernel< uint8_t >( filter1, size, byteRef[ i ] );
            t2 = _testSDKKernel< float >( filter2, size, floatRef[ i ] );
            std::cout << std::setw(7) << filter1->GetNumberOfThreads() << ','
                      << std::setw(14) << size*size*size / 1024.f / 1024.f / t1
                      << ',' << std::setw(15)
                      << size*size*size / 1024.f / 1024.f / t2 << std::endl;
        }
    }
}
