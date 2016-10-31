
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

#include "volumeHandler.h"

namespace fivox
{
VolumeHandler::VolumeHandler( const size_t size, const vmml::Vector3f& extent )
    : _size( size )
    , _extent( extent )
{}

FloatVolume::RegionType
VolumeHandler::computeRegion( const Vector2ui& decompose ) const
{
    const size_t begin = float( _size ) / float( decompose[1] ) *
                         float( decompose[0] );
    const size_t end = size_t( float( _size ) / float( decompose[1] ) *
                               float( decompose[0] + 1 )) - 1;

    const size_t maxExtentIndex = _extent.find_max_index();

    FloatVolume::IndexType vIndex;
    vIndex.Fill( 0 );
    vIndex[ maxExtentIndex ] = begin;

    FloatVolume::SizeType vSize;
    vSize[ maxExtentIndex ] = end - begin + 1;
    for( size_t i = 0; i < 3; ++i )
    {
        if( i != maxExtentIndex )
            vSize[i] = _size * _extent[i] / _extent.find_max();
    }
    return FloatVolume::RegionType( vIndex, vSize );
}

FloatVolume::SpacingType VolumeHandler::computeSpacing() const
{
    FloatVolume::SpacingType spacing;
    spacing.Fill( _extent.find_max() / float( _size ));

    return spacing;
}

FloatVolume::PointType
VolumeHandler::computeOrigin( const Vector3f& center ) const
{
    const Vector3f& position( center - _extent * 0.5f );

    typename FloatVolume::PointType origin;
    origin[0] = position[0];
    origin[1] = position[1];
    origin[2] = position[2];

    return origin;
}

}
