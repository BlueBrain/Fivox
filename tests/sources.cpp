
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Eyescale Software GmbH nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define BOOST_TEST_MODULE Sources

#include "test.h"
#include <fivox/compartmentLoader.h>
#include <fivox/eventFunctor.h>
#include <fivox/helpers.h>
#include <fivox/imageSource.h>
#ifdef FIVOX_USE_LFP
#  include <fivox/lfp/lfpFunctor.h>
#endif
#include <fivox/somaLoader.h>
#include <fivox/spikeLoader.h>
#include <fivox/synapseLoader.h>
#include <fivox/uriHandler.h>
#include <brion/spikeReport.h>

#include <BBP/TestDatasets.h>

#include <itkImageFileWriter.h>
#include <itkStatisticsImageFilter.h>
#include <itkTimeProbe.h>

#include <lunchbox/sleep.h>
#include <lunchbox/pluginRegisterer.h>

#include <iomanip>

#define STARTUP_DELAY 250
#define WRITE_DELAY 100

namespace
{

const std::string _monsteerPluginScheme( "monsteer" );
const size_t _minResolution = 8;

template< typename Image >
inline float _testKernel(
        fivox::ImageSourcePtr< Image > filter, const size_t size,
        const float expectedValue, const vmml::Vector2ui& range )
{
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    // set up size and origin for loaded data
    fivox::EventSourcePtr source = filter->getEventSource();
    source->setTime( 0.f );
    const fivox::AABBf& bbox = source->getBoundingBox();
    const fivox::Vector3f& position = bbox.getMin();
    const float extent = bbox.getSize().find_max();
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
    fivox::EventSource& sourceRef = *source;
    os << size << '_' << typeid( sourceRef ).name() << '_'
       << typeid( Image ).name() << ".mhd";

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

    void testSource( const fivox::URI& uri,
                     const float byteRef, const float floatRef,
                     const vmml::Vector2ui& rangeRef )
    {
        const fivox::URIHandler params( uri );
        auto filter1 = params.newImageSource< fivox::ByteVolume >();
        auto filter2 = params.newImageSource< fivox::FloatVolume >();

        std::cout << "   Size, " << uri << "," << std::endl;
        for( size_t j = _minResolution; j <= maxSize; j = j << 1 )
        {
            const float t1 =
                _testKernel< fivox::ByteVolume >( filter1, j, byteRef, rangeRef );
            const float t2 =
                _testKernel< fivox::FloatVolume >( filter2, j, floatRef, rangeRef );
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

            float t1 =
                _testKernel< fivox::ByteVolume >( filter1, size, byteRef, rangeRef );
            float t2 =
                _testKernel< fivox::FloatVolume >( filter2, size, floatRef, rangeRef );
            std::cout << std::setw(7) << filter1->GetNumberOfThreads() << ','
                      << std::setw(14) << size*size*size / 1024.f / 1024.f / t1
                      << ',' << std::setw(15)
                      << size*size*size / 1024.f / 1024.f / t2 << std::endl;

            filter1->SetNumberOfThreads( maxThreads / j );
            filter2->SetNumberOfThreads( maxThreads / j );
            t1 = _testKernel< fivox::ByteVolume >( filter1, size, byteRef, rangeRef );
            t2 = _testKernel< fivox::FloatVolume >( filter2, size, floatRef, rangeRef );
            std::cout << std::setw(7) << filter1->GetNumberOfThreads() << ','
                      << std::setw(14) << size*size*size / 1024.f / 1024.f / t1
                      << ',' << std::setw(15)
                      << size*size*size / 1024.f / 1024.f / t2 << std::endl;
        }
    }
};
}

BOOST_FIXTURE_TEST_SUITE( sources, SourcesFixture )

BOOST_AUTO_TEST_CASE( fivoxVoltages_source )
{
    // Compartment report 'voltages' (binary) contains timestamps
    // between 0 and 100 with a Dt=0.1 => data range is 0.0 to 10.0 ms
    testSource( fivox::URI( "fivoxCompartments://" ), 5.455078125f,
                -0.062685228640475543, vmml::Vector2ui( 0, 100 ));
}

BOOST_AUTO_TEST_CASE( fivoxSomas_source )
{
    // Soma report 'somas' (binary) contains timestamps
    // between 0 and 100 with a Dt=0.1 => data range is 0.0 to 10.0 ms
    testSource( fivox::URI( "fivoxSomas://" ), 0.f,
                -0.0016181814135052264, vmml::Vector2ui( 0, 100 ));
}

#ifdef FIVOX_USE_LFP
BOOST_AUTO_TEST_CASE( fivoxLFP_source )
{
    // Compartment currents report 'currents' (binary) contains timestamps
    // between 0 and 100 with a Dt=0.1 => data range is 0.0 to 10.0 ms
    testSource( fivox::URI( "fivoxCompartments://?functor=lfp" ), 0.f,
                3.3634767649011466e-09f, vmml::Vector2ui( 0, 100 ));
}
#endif

BOOST_AUTO_TEST_CASE( fivoxSpikes_source )
{
    // Spikes report timestamps range between 0.725 and 9.975 ms
    testSource( fivox::URI( "fivoxSpikes://?duration=1&dt=1&target=Column" ),
                0.005859375f, 0.005859375f, vmml::Vector2ui( 0, 9 ));
}

BOOST_AUTO_TEST_CASE( fivoxSynapses_source )
{
    // Synapse reports don't have time support and return a 1-frame interval
    testSource( fivox::URI( "fivoxSynapses://?target=Column" ), 7.42578125f,
                437.92578125f, vmml::Vector2ui( 0, 1 ));
}

BOOST_AUTO_TEST_CASE( fivoxVSD_source )
{
    testSource( fivox::URI( "fivoxVSD://?target=allmini50" ),
                12.703125, -85293.598821282387f, vmml::Vector2ui( 0, 100 ));
}

BOOST_AUTO_TEST_SUITE_END()

#if FIVOX_USE_MONSTEER

BOOST_AUTO_TEST_CASE( fivoxSpikes_stream_source_frame_range )
{
    brion::SpikeReport spikeWriter(
        servus::URI( _monsteerPluginScheme + "://127.0.0.1" ),
        brion::MODE_WRITE );
    servus::URI uri = spikeWriter.getURI();
    uri.setScheme( _monsteerPluginScheme );

    fivox::URIHandler params(
        "fivoxspikes://?dt=0.1&duration=1&spikes=" + std::to_string( uri ));
    auto filter = params.newImageSource< float >();
    fivox::EventSourcePtr source = filter->getEventSource();

    lunchbox::sleep( STARTUP_DELAY );

    brion::Spikes spikes;
    for( uint32_t i = 0; i <= 50; ++i )
        spikes.insert( std::make_pair( i / 100.0f, i ));
    spikeWriter.writeSpikes( spikes );

    // Time range: [0, 0.5](ms)
    // Since duration=1(ms), no full frames are available: [0,0)
    BOOST_CHECK_EQUAL( source->getFrameRange(), fivox::Vector2ui( 0, 0 ));

    spikes.clear();
    for( uint32_t i = 51; i <= 100; ++i )
        spikes.insert( std::make_pair( i / 100.0f, i ));
    spikeWriter.writeSpikes( spikes );

    lunchbox::sleep( WRITE_DELAY );

    // The very last spike is not digested by the internal SpikeReportReader
    // because to provide complete data inside a time window it cannot advance
    // past time t_max until one spike with t > t_max arrives.
    BOOST_CHECK_EQUAL( source->getFrameRange(), fivox::Vector2ui( 0, 0 ));

    spikes.clear();
    for( uint32_t i = 101; i <= 120; ++i )
        spikes.insert( std::make_pair( i / 100.0f, i ));
    spikeWriter.writeSpikes( spikes );

    lunchbox::sleep( WRITE_DELAY );
    BOOST_CHECK_EQUAL( source->getFrameRange(), fivox::Vector2ui( 0, 1 ));

    spikes.clear();
    for( uint32_t i = 121; i <= 150; ++i )
        spikes.insert( std::make_pair( i / 100.0f, i ));
    spikeWriter.writeSpikes( spikes );
    lunchbox::sleep( WRITE_DELAY );
    // The time window [1.4, 1.5) is still not complete for the reason explained
    // above.
    BOOST_CHECK_EQUAL( source->getFrameRange(), fivox::Vector2ui( 0, 4 ));

    spikes.clear();
    for( uint32_t i = 151; i <= 200; ++i )
        spikes.insert( std::make_pair( i / 100.0f, i ));
    spikeWriter.writeSpikes( spikes );
    // After closing the report all spikes are made available, even for the
    // time window.
    spikeWriter.close();
    lunchbox::sleep( WRITE_DELAY );
    // Time range: [0, 2.0](ms)
    // Since duration=1(ms) and dt=0.1(ms), 11 full frames are available: [0,11)
    BOOST_CHECK_EQUAL( source->getFrameRange(), fivox::Vector2ui( 0, 11 ));
}

#endif
