/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "event.h"
#include "testLoader.h"
#include "uriHandler.h"

#ifdef final
#  undef final
#endif

namespace fivox
{

class TestLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _magnitude( params.getMagnitude( ))
    {
        for( uint8_t y = 0; y < 10; ++y )
            output.add( Event( Vector3f( 0.f, y * 10.f, 0.f ),
                               VALUE_UNSET, 1.f ));

        // WAR: set the extent of the generated volume (voxelize tool)
        output.add( Event( Vector3f( -50.f, 150.f, -50.f ), VALUE_UNSET ));
        output.add( Event( Vector3f( 50.f, -50.f, 50.f ), VALUE_UNSET ));
    }

    ssize_t load( const float time )
    {
        const size_t numEvents = _output.getEvents().size() - 2;
        for( size_t i = 0; i < numEvents; ++i )
            _output[i].value = (i + 1 + time) * _magnitude;

        return numEvents;
    }

    EventSource& _output;
    const float _magnitude;
};

TestLoader::TestLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new TestLoader::Impl( *this, params ))
{}

TestLoader::~TestLoader()
{}

Vector2f TestLoader::_getTimeRange() const
{
    return Vector2f( 0.f, 100.f );
}

ssize_t TestLoader::_load( const float time )
{
    return _impl->load( time );
}

}
