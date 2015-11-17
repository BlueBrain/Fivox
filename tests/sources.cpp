
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
    const size_t size, const float expectedValue, const vmml::Vector2ui& range )
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
    BOOST_CHECK_EQUAL( source->getFrameRange(), range );

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

struct SourcesFixture
{
    SourcesFixture()
        : argv( boost::unit_test::framework::master_test_suite().argv)
        , unitTest( std::string( argv[0] ).find( "perf-" ) ==
                    std::string::npos )
        , maxSize( unitTest ? _minResolution : 1024 )
        , maxThreads( unitTest ? 0 : 16 )
    {
        std::cout.setf( std::ios::right, std::ios::adjustfield );
        std::cout.precision( 5 );
        std::cout << "    Test, byte MVox/sec, float MVox/sec" << std::endl;
    }

    char**const argv;
    const bool unitTest;
    const size_t maxSize;
    const size_t maxThreads;

    void testSource( const std::string& uri,
                     const float byteRef, const float floatRef,
                     const vmml::Vector2ui& rangeRef )
    {
        const fivox::URIHandler params( uri );
        auto filter1 = params.newImageSource< uint8_t >();
        auto filter2 = params.newImageSource< float >();

        std::cout << "   Size, " << uri << "," << std::endl;
        for( size_t j = _minResolution; j <= maxSize; j = j << 1 )
        {
            const float t1 = _testSDKKernel< uint8_t >( filter1, j,
                                                        byteRef, rangeRef );
            const float t2 = _testSDKKernel< float >( filter2, j,
                                                      floatRef, rangeRef );
            std::cout << std::setw( 8 ) << j << ',' << std::setw(14)
                      << j*j*j / 1024.f / 1024.f / t1 << ',' << std::setw(15)
                      << j*j*j / 1024.f / 1024.f / t2 << std::endl;
        }

        if( maxThreads )
            std::cout << "Threads, " << uri << "," << std::endl;
        const size_t size = maxSize >> 2;
        for( size_t j = 1; j <= maxThreads; j = j << 1 )
        {
            filter1->SetNumberOfThreads( maxThreads * j );
            filter2->SetNumberOfThreads( maxThreads * j );

            float t1 = _testSDKKernel< uint8_t >( filter1, size,
                                                  byteRef, rangeRef );
            float t2 = _testSDKKernel< float >( filter2, size,
                                                floatRef, rangeRef );
            std::cout << std::setw(7) << filter1->GetNumberOfThreads() << ','
                      << std::setw(14) << size*size*size / 1024.f / 1024.f / t1
                      << ',' << std::setw(15)
                      << size*size*size / 1024.f / 1024.f / t2 << std::endl;

            filter1->SetNumberOfThreads( maxThreads / j );
            filter2->SetNumberOfThreads( maxThreads / j );
            t1 = _testSDKKernel< uint8_t >( filter1, size, byteRef, rangeRef );
            t2 = _testSDKKernel< float >( filter2, size, floatRef, rangeRef );
            std::cout << std::setw(7) << filter1->GetNumberOfThreads() << ','
                      << std::setw(14) << size*size*size / 1024.f / 1024.f / t1
                      << ',' << std::setw(15)
                      << size*size*size / 1024.f / 1024.f / t2 << std::endl;
        }
    }
};
}

BOOST_FIXTURE_TEST_SUITE( sources, SourcesFixture )

BOOST_AUTO_TEST_CASE( fivox_source )
{
    // Compartement report 'allvoltage' (binary) contains timestamps
    // between 0 and 1000 with a Dt=0.1 => data range is 0.0 to 100.0 ms
    testSource( "fivox://", 3.017578125, 0.0127238630805806,
                vmml::Vector2ui( 0, 1000 ));
}

BOOST_AUTO_TEST_CASE( fivoxSomas_source )
{
    // Soma report 'voltage' (binary) contains timestamps
    // between 0 and 1000 with a Dt=0.1 => data range is 0.0 to 100.0 ms
    testSource( "fivoxSomas://", 0.521484375f, 0.0034715187880252074f,
                vmml::Vector2ui( 0, 1000 ));
}

BOOST_AUTO_TEST_CASE( fivoxSpikes_source )
{
    // Spikes report timestamps range between 0.25 and 99.95 ms
    // Better, but not always available:
    // "fivoxSpikes:///gpfs/bbp.cscs.ch/home/nachbaur/BlueConfig_3m",
    testSource( "fivoxSpikes://", 0.7421875f, 0.0029296876164153218f,
                vmml::Vector2ui( 10, 3998 ));

}

BOOST_AUTO_TEST_CASE( fivoxSynapses_source )
{
    // Synapse reports don't have time support and return a 1-frame interval
    // Better, but not always available data:
    // "fivoxSynapses:///gpfs/bbp.cscs.ch/home/nachbaur/BlueConfig_3m,target=L23_DBC",
    testSource( "fivoxSynapses://", 0.03515625f, 0.00017834029219887526f,
                vmml::Vector2ui( 0, 1 ));
}

BOOST_AUTO_TEST_SUITE_END()
