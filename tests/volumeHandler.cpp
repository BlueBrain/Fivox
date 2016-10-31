
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

#define BOOST_TEST_MODULE VolumeHandler

#include "test.h"
#include <fivox/volumeHandler.h>

BOOST_AUTO_TEST_CASE( VolumeHandlerRegion )
{
    fivox::FloatVolume::RegionType region;
    fivox::FloatVolume::IndexType expectedIndex;
    fivox::FloatVolume::SizeType expectedSize;

    // even number of voxels
    fivox::VolumeHandler volumeHandler( 100, fivox::Vector3f( 50, 100, 42 ));

    // full volume region
    region = volumeHandler.computeRegion( fivox::Vector2ui( 0, 1 ));
    expectedIndex[0] = 0; expectedIndex[1] = 0; expectedIndex[2] = 0;
    expectedSize[0] = 50; expectedSize[1] = 100; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );

    // even number of regions
    region = volumeHandler.computeRegion( fivox::Vector2ui( 1, 2 ));
    expectedIndex[0] = 0; expectedIndex[1] = 50; expectedIndex[2] = 0;
    expectedSize[0] = 50; expectedSize[1] = 50; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );

    // odd number of regions
    region = volumeHandler.computeRegion( fivox::Vector2ui( 1, 3 ));
    expectedIndex[0] = 0; expectedIndex[1] = 33; expectedIndex[2] = 0;
    expectedSize[0] = 50; expectedSize[1] = 33; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );

    region = volumeHandler.computeRegion( fivox::Vector2ui( 2, 3 ));
    expectedIndex[0] = 0; expectedIndex[1] = 66; expectedIndex[2] = 0;
    expectedSize[0] = 50; expectedSize[1] = 34; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );

    // odd number of voxels
    volumeHandler.setSize( 101 );
    volumeHandler.setExtent( fivox::Vector3f( 100, 50, 42 ));

    // even number of regions
    region = volumeHandler.computeRegion( fivox::Vector2ui( 1, 2 ));
    expectedIndex[0] = 50; expectedIndex[1] = 0; expectedIndex[2] = 0;
    expectedSize[0] = 51; expectedSize[1] = 50; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );

    // odd number of regions
    region = volumeHandler.computeRegion( fivox::Vector2ui( 0, 3 ));
    expectedIndex[0] = 0; expectedIndex[1] = 0; expectedIndex[2] = 0;
    expectedSize[0] = 33; expectedSize[1] = 50; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );

    region = volumeHandler.computeRegion( fivox::Vector2ui( 2, 3 ));
    expectedIndex[0] = 67; expectedIndex[1] = 0; expectedIndex[2] = 0;
    expectedSize[0] = 34; expectedSize[1] = 50; expectedSize[2] = 42;
    BOOST_CHECK_EQUAL( region.GetIndex(), expectedIndex );
    BOOST_CHECK_EQUAL( region.GetSize(), expectedSize );
}

BOOST_AUTO_TEST_CASE( VolumeHandlerSpacing )
{
    fivox::FloatVolume::SpacingType expectedSpacing;
    fivox::VolumeHandler volumeHandler( 100, fivox::Vector3f( 50, 100, 42 ));
    expectedSpacing.Fill( 1 );
    BOOST_CHECK_EQUAL( volumeHandler.computeSpacing(), expectedSpacing );

    volumeHandler.setSize( 200 );
    expectedSpacing.Fill( 0.5 );
    BOOST_CHECK_EQUAL( volumeHandler.computeSpacing(), expectedSpacing );

    volumeHandler.setSize( 50 );
    expectedSpacing.Fill( 2 );
    BOOST_CHECK_EQUAL( volumeHandler.computeSpacing(), expectedSpacing );
}

BOOST_AUTO_TEST_CASE( VolumeHandlerOrigin )
{
    fivox::FloatVolume::PointType expectedOrigin;
    fivox::VolumeHandler volumeHandler( 100, fivox::Vector3f( 50, 100, 42 ));

    expectedOrigin[0] = -25; expectedOrigin[1] = -50; expectedOrigin[2] = -21;
    BOOST_CHECK_EQUAL( volumeHandler.computeOrigin( fivox::Vector3f( 0, 0, 0 )),
                       expectedOrigin );

    expectedOrigin[0] = 75; expectedOrigin[1] = 50; expectedOrigin[2] = 79;
    BOOST_CHECK_EQUAL(
                volumeHandler.computeOrigin( fivox::Vector3f( 100, 100, 100 )),
                expectedOrigin );

    expectedOrigin[0] = -75; expectedOrigin[1] = -100; expectedOrigin[2] = -71;
    BOOST_CHECK_EQUAL(
                volumeHandler.computeOrigin( fivox::Vector3f( -50, -50, -50 )),
                expectedOrigin );
}
