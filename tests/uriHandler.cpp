
/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */

#define BOOST_TEST_MODULE URIHandler

#include "test.h"
#include <fivox/uriHandler.h>

#ifdef FIVOX_USE_BBPTESTDATA
#  include <BBP/TestDatasets.h>
#  include <lunchbox/file.h>
#endif

BOOST_AUTO_TEST_CASE(URIHandlerCompartments)
{
    const fivox::URIHandler handler( "fivox://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::COMPARTMENTS );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "Layer1" );
    BOOST_CHECK_EQUAL( handler.getReport(), "allvoltage" );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
    BOOST_CHECK_EQUAL( handler.getReport(), "voltage" );
#endif
    BOOST_CHECK_EQUAL( handler.getTarget( "foo" ), "foo" );
    BOOST_CHECK_EQUAL( handler.getDt(), -1.f );
    BOOST_CHECK_EQUAL( handler.getDuration(), 10.0f );

    const fivox::URIHandler params1(
        "fivoxcompartment:///path/to/BlueConfig?report=simulation,dt=0.2,target=Column" );
    BOOST_CHECK_EQUAL( params1.getConfig(), "/path/to/BlueConfig" );
    BOOST_CHECK_EQUAL( params1.getTarget( "" ), "Column" );
    BOOST_CHECK_EQUAL( params1.getTarget( "foo" ), "Column" );
    BOOST_CHECK_EQUAL( params1.getReport(), "simulation" );
    BOOST_CHECK_EQUAL( params1.getDt(), 0.2f );

    const fivox::URIHandler params2(
        "fivoxcompartment:///path/to/BlueConfig?target=First#Second" );
    BOOST_CHECK_EQUAL( params2.getConfig(), "/path/to/BlueConfig" );
    BOOST_CHECK_EQUAL( params2.getTarget( "" ), "First" );

    const fivox::URIHandler params3(
        "fivoxcompartment:///path/to/BlueConfig?report=simulation#Second" );
    BOOST_CHECK_EQUAL( params3.getConfig(), "/path/to/BlueConfig" );
    BOOST_CHECK_EQUAL( params3.getTarget( "" ), "Second" );
    BOOST_CHECK_EQUAL( params3.getReport(), "simulation" );
}

BOOST_AUTO_TEST_CASE(URIHandlerSoma)
{
    const fivox::URIHandler handler( "fivoxsomas://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::SOMAS );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "Layer1" );
    BOOST_CHECK_EQUAL( handler.getReport(), "voltage" );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
    BOOST_CHECK_EQUAL( handler.getReport(), "soma" );
#endif
}

BOOST_AUTO_TEST_CASE(URIHandlerSpikes)
{
    const fivox::URIHandler handler( "fivoxspikes://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::SPIKES );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "Column" );
    BOOST_CHECK_EQUAL( handler.getReport(), "allvoltage" );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
    BOOST_CHECK_EQUAL( handler.getReport(), "voltage" );
#endif
}

BOOST_AUTO_TEST_CASE(URIHandlerSynapses)
{
    const fivox::URIHandler handler( "fivoxsynapses://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::SYNAPSES );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "Column" );
    BOOST_CHECK_EQUAL( handler.getReport(), "allvoltage" );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
    BOOST_CHECK_EQUAL( handler.getReport(), "voltage" );
#endif
}

BOOST_AUTO_TEST_CASE(URIHandlerVSD)
{
    const fivox::URIHandler handler( "fivoxvsd://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::VSD );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), lunchbox::getExecutablePath()
                       + "/../share/Fivox/configs/BlueConfigVSD" );
#endif
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "Layer1" );
    BOOST_CHECK_EQUAL( handler.getReport(), "voltage" );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
    BOOST_CHECK_EQUAL( handler.getReport(), "soma" );
#endif
}
