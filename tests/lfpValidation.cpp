
/* Copyright (c) 2016, EPFL/Blue Brain Project
 *                          grigori.chevtchenko@epfl.ch
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

#define BOOST_TEST_MODULE lfpValidation

#include "test.h"
#include <fivox/eventSource.h>
#include <fivox/imageSource.h>
#include <fivox/eventFunctor.h>
#include <fivox/uriHandler.h>

#include <itkImageFileWriter.h>

// This test compares 2 values computed by our LFP implementation with 2 values
// computed by hand in accordance to the electro-magnetic theory. The parameters
// of the computation are:
//     -loader = fivoxtest ( contains 7 events and 100 frames )
//     -cutoff = 100 micrometers
//     -resolution = 1 voxel/micrometer
//     -frame = 90

const float expectedValue = 2.36366f;
const float expectedValue2 = 2.81135f;

BOOST_AUTO_TEST_CASE( LfpValidation )
{
    const fivox::URIHandler params( fivox::URI(
                   "fivox://?resolution=1&cutoff=100&functor=lfp&extend=100" ));
    auto volumeSource = params.newImageSource< fivox::FloatVolume >();

    typedef fivox::FloatVolume Image;
    itk::Size<3> size = {{205, 240, 204}};
    Image::RegionType region;
    region.SetSize( size );

    fivox::FloatVolume::Pointer volume = volumeSource->GetOutput();
    volume->SetRegions( region );

    // set up size and origin for loaded data
    fivox::EventSourcePtr source = volumeSource->getEventSource();
    source->setFrame( 90u );
    const fivox::AABBf& bbox = source->getBoundingBox();
    const fivox::Vector3f& extent( bbox.getSize() +
                                   params.getExtendDistance() * 2.f );
    const fivox::Vector3f& position = bbox.getCenter() - extent * 0.5;

    Image::SpacingType spacing;
    spacing.Fill( 1.0 / params.getResolution( ));
    volume->SetSpacing( spacing );

    Image::PointType origin;
    origin[0] = position[0];
    origin[1] = position[1];
    origin[2] = position[2];
    volume->SetOrigin( origin );
    volumeSource->Modified();
    volumeSource->Update();

    BOOST_CHECK_CLOSE( volume->GetPixel({{ 150, 150, 150 }}), expectedValue, 0.001f/*%*/ );
    BOOST_CHECK_CLOSE( volume->GetPixel({{ 50, 80, 80 }}), expectedValue2, 0.001f/*%*/ );
}
