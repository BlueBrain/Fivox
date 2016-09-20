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
#include "uriHandler.h"

#include <brain/brain.h>

#include <future>


namespace fivox
{

class SynapseLoader::Impl
{
public:
    Impl( EventSource& output, const URIHandler& params )
        : _output( output )
        , _circuit( params.getConfig( ))
        , _preGIDs( params.getPreGIDs( ))
        , _postGIDs( params.getGIDs( ))
        , _synapses( _loadSynapseStream( ))
        , _numChunks( _synapses.getRemaining( ))
    {
        if( !params.getReferenceVolume().empty( ))
            return;

        // compute circuit bounding box as we don't have any synapses at this
        // point
        const auto& gids = _circuit.getGIDs();
        const brion::Vector3fs& positions = _circuit.getPositions( gids );

        AABBf bbox;
        for( const auto& position : positions )
            bbox.merge( position );
        _output.setBoundingBox( bbox );
    }

    brain::SynapsesStream _loadSynapseStream()
    {
        if( _preGIDs.empty( ))
            return _circuit.getAfferentSynapses( _postGIDs,
                                            brain::SynapsePrefetch::positions );

        return _circuit.getProjectedSynapses( _preGIDs, _postGIDs,
                                            brain::SynapsePrefetch::positions );
    }

    ssize_t load( const size_t /*chunkIndex*/, const size_t numChunks )
    {
        // prefetching the next chunk (instead of synchronously waiting here)
        // turned out to be slower in real life...
        const brain::Synapses synapses = _synapses.read( numChunks ).get();
        if( _synapses.eos( ))
            _synapses = _loadSynapseStream();
        _output.resize( synapses.size( ));
        const float* __restrict__ posx = synapses.preSurfaceXPositions();
        const float* __restrict__ posy = synapses.preSurfaceYPositions();
        const float* __restrict__ posz = synapses.preSurfaceZPositions();
        for( size_t i = 0; i < synapses.size(); ++i )
            _output.update( i, Vector3f( posx[i], posy[i], posz[i] ),
                            /*radius*/ 0.f, /*value*/ 1.f );

        return synapses.size();
    }

    EventSource& _output;
    const brain::Circuit _circuit;
    const brain::GIDSet _preGIDs;
    const brain::GIDSet _postGIDs;
    brain::SynapsesStream _synapses;
    const size_t _numChunks;
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

ssize_t SynapseLoader::_load( const size_t chunkIndex, const size_t numChunks )
{
    return _impl->load( chunkIndex, numChunks );
}

size_t SynapseLoader::_getNumChunks() const
{
    return _impl->_numChunks;
}

}
