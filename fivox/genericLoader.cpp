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

#include "genericLoader.h"
#include "uriHandler.h"

#include <lunchbox/log.h>

#ifdef final
#  undef final
#endif

namespace fivox
{

class GenericLoader::Impl
{
public:
    explicit Impl( EventSource& output, const URIHandler& params )
        : _output( output )
        , _file( params.getConfigPath( ))
    {
        if( _file.empty( ))
        {
            _output.resize( 7 );

            for( uint8_t y = 0; y < 5; ++y )
                _output.update( y, Vector3f( 0.f, y * 10.f, 0.f ), 1.f );

            _output.update( 5, Vector3f( 3.f, 5.f, 4.f ), 1.f );
            _output.update( 6, Vector3f( 5.f, 2.f, 1.f ), 1.f );

            return;
        }

        _output.read( _file );
    }

    ssize_t load()
    {
        const size_t numEvents = _output.getNumEvents();
        for( size_t i = 0; i < numEvents; ++i )
            _output[i] = (i + 1 + _output.getCurrentTime( ));

        return numEvents;
    }

    EventSource& _output;
    const std::string& _file;
};

GenericLoader::GenericLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new GenericLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( 1.f );
}

GenericLoader::~GenericLoader()
{}

Vector2f GenericLoader::_getTimeRange() const
{
    return Vector2f( 0.f, 100.f );
}

ssize_t GenericLoader::_load( const size_t /*chunkIndex*/,
                           const size_t /*numChunks*/ )
{
    return _impl->load();
}

}
