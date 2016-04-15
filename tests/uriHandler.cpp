
/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
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
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::TYPE_COMPARTMENTS );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
#endif
    BOOST_CHECK_EQUAL( handler.getReport(), "voltages" );
    BOOST_CHECK_EQUAL( handler.getTarget( "foo" ), "foo" );
    BOOST_CHECK_EQUAL( handler.getDt(), -1.f );
    BOOST_CHECK_EQUAL( handler.getDuration(), 10.0f );

    const fivox::URIHandler params1(
        "fivoxcompartment:///path/to/BlueConfig?report=simulation&dt=0.2&target=Column" );
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
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::TYPE_SOMAS );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
#endif
    BOOST_CHECK_EQUAL( handler.getReport(), "somas" );
}

BOOST_AUTO_TEST_CASE(URIHandlerSpikes)
{
    const fivox::URIHandler handler( "fivoxspikes://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::TYPE_SPIKES );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
#endif
    BOOST_CHECK_EQUAL( handler.getReport(), "voltages" );
}

BOOST_AUTO_TEST_CASE(URIHandlerSynapses)
{
    const fivox::URIHandler handler( "fivoxsynapses://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::TYPE_SYNAPSES );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
#endif
    BOOST_CHECK_EQUAL( handler.getReport(), "voltages" );
}

BOOST_AUTO_TEST_CASE(URIHandlerVSD)
{
    const fivox::URIHandler handler( "fivoxvsd://" );
    BOOST_CHECK_EQUAL( handler.getType(), fivox::VolumeType::TYPE_VSD );
#ifdef FIVOX_USE_BBPTESTDATA
    BOOST_CHECK_EQUAL( handler.getConfig(), BBP_TEST_BLUECONFIG );
#else
    BOOST_CHECK_EQUAL( handler.getTarget( "" ), "" );
#endif
    BOOST_CHECK_EQUAL( handler.getReport(), "voltages" );
}
