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

#include <lunchbox/log.h>

#ifdef final
#  undef final
#endif

namespace fivox
{

class TestLoader::Impl
{
public:
    Impl( EventSource& output )
        : _output( output )
    {
        for( uint8_t y = 0; y < 5; ++y )
            output.add( Event( Vector3f( 0.f, y * 10.f, 0.f ),
                               VALUE_UNSET, 1.f ));

        output.add( Event( Vector3f( 3.f, 5.f, 4.f ),
                           VALUE_UNSET, 1.f ));

        output.add( Event( Vector3f( 5.f, 2.f, 1.f ),
                           VALUE_UNSET, 1.f ));
    }

    ssize_t load( const float time )
    {
        const size_t numEvents = _output.getEvents().size();
        for( size_t i = 0; i < numEvents; ++i )
            _output[i].value = (i + 1 + time);

        return numEvents;
    }

    EventSource& _output;
};

TestLoader::TestLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new TestLoader::Impl( *this ))
{
    if( getDt() < 0.f )
        setDt( 1.f );
}

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
