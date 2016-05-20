/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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

#include "synapseLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <brion/brion.h>
#include <lunchbox/os.h>
#include <lunchbox/memoryMap.h>
#include <boost/progress.hpp>
#ifdef final
#  undef final
#endif

namespace fivox
{
class SynapseLoader::Impl
{
public:
    Impl( EventSource& output, const URIHandler& params )
        : _output( output )
    {
        const auto& gids = params.getGIDs();

        LBINFO << "Loading synapses for " << gids.size() << " cells..."
               << std::endl;
        boost::progress_display progress( gids.size( ));
        const brion::Synapse synapses( params.getConfig().getSynapseSource().getPath() +
                                       "/nrn_positions.h5" );

        for( const uint32_t gid : gids )
        {
            const brion::SynapseMatrix& data =
                synapses.read( gid, brion::SYNAPSE_PRESYNAPTIC_SURFACE_X |
                                    brion::SYNAPSE_PRESYNAPTIC_SURFACE_Y |
                                    brion::SYNAPSE_PRESYNAPTIC_SURFACE_Z );
            for( size_t i = 0; i < data.shape()[0]; ++i )
                _output.add( Event( Vector3f( data[i][0], data[i][1],
                                              data[i][2] ), 1.f ));
            ++progress;
        }
    }

private:
    EventSource& _output;
};

SynapseLoader::SynapseLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new Impl( *this, params ))
{
    setDt( 1.f );
}

SynapseLoader::~SynapseLoader()
{}

Vector2f SynapseLoader::_getTimeRange() const
{
    return Vector2f( 0.f, 1.f );
}

ssize_t SynapseLoader::_load( const float )
{
    return 0;
}

}
