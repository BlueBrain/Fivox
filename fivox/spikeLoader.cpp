/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
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

#include "spikeLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <brain/brain.h>
#include <brion/brion.h>
#include <brain/spikeReportReader.h>
#include <brain/spikes.h>
#include <lunchbox/os.h>
#include <lunchbox/lock.h>
#include <lunchbox/scopedMutex.h>
#include <lunchbox/memoryMap.h>

#ifdef final
#  undef final
#endif

using boost::lexical_cast;

namespace fivox
{
class SpikeLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _config( params.getConfig( ))
        , _duration( params.getDuration( ))
        , _spikesStart( 0.f )
        , _spikesEnd( 0.f )
    {
        const brain::Circuit circuit( _config );
        const brion::GIDSet& gids = _config.parseTarget(
                                params.getTarget( _config.getCircuitTarget( )));
        if( gids.empty( ))
            LBTHROW( std::runtime_error(
                         "No GIDs found for default circuit target in "+
                         params.getConfig( )));

        const brion::Vector3fs& positions = circuit.getPositions( gids );

        size_t i = 0;
        _gidIndex.resize( *gids.rbegin() + 1 );
        for( const uint32_t gid: gids )
        {
            _output.add( Event( positions[ i ], VALUE_UNSET ));
            _gidIndex[gid] = i++;
        }
        _spikesPerNeuron.resize( gids.size( ));
        const std::string& spikePath = params.getSpikes();
        _loadSpikes( spikePath.empty() ? _config.getSpikeSource() :
                                         brion::URI( spikePath ));

    }

    void updateTimeRange()
    {
        // Streaming in progress; Only report fully finished frames.
        if( _spikesReader && !_spikesReader->hasEnded( ))
        {
            lunchbox::ScopedWrite mutex( _getSpikesLock );
            const brain::Spikes& spikes = _spikesReader->getSpikes();
            if( !spikes.empty( ))
                // don't update _spikesStart to calculate absolute frame numbers
                // see https://bbpcode.epfl.ch/code/#/c/19337
                _spikesEnd = spikes.getEndTime();
        }
    }

    void _loadSpikes( const brion::URI& spikes )
    {
        if( _loadBinarySpikes( spikes.getPath( )))
            return;

        LBINFO << "No valid binary .spikes file found, loading from .dat..."
               << std::endl;
        _spikesReader.reset( new brain::SpikeReportReader( spikes ));
        _spikesStart = _spikesReader->isStream() ? 0.f :
                                                _spikesReader->getStartTime();
        _spikesEnd = _spikesReader->isStream() ? 0.f :
                                                _spikesReader->getEndTime();
    }

    bool _loadBinarySpikes( const std::string& spikes )
    {
        std::unique_ptr< lunchbox::MemoryMap > spikesFile
                ( new lunchbox::MemoryMap( spikes ));
        const size_t size = spikesFile->getSize();
        if(( size % sizeof( uint32_t )) != 0 )
            return false;
        const size_t nElems = size / sizeof( uint32_t );
        const uint32_t* iData = spikesFile->getAddress< uint32_t >();
        size_t index = 0;

        const uint32_t magic = 0xf0a;
        const uint32_t version = 1;
        if( index >= nElems || iData[ index++ ] != magic )
            return false;
        if( index >= nElems || iData[ index++ ] != version )
            return false;

        _spikesFile = std::move( spikesFile );
        const float* fData = _spikesFile->getAddress< float >();
        _spikesStart = fData[index]; // first spike timestamp after header
        _spikesEnd = fData[nElems - 2]; // last spike timestamp
        return true;
    }

    ssize_t load( const float start )
    {
        lunchbox::setZero( _spikesPerNeuron.data(),
                           _spikesPerNeuron.size() * sizeof(size_t));

        const float end = start + _duration;
        const size_t numSpikes = _spikesFile ? _loadSpikesFast( start, end )
                                             : _loadSpikesSlow( start, end );

        for( size_t i = 0; i < _spikesPerNeuron.size(); ++i )
            _output[i].value = _spikesPerNeuron[i] ? _spikesPerNeuron[i]
                                                   : VALUE_UNSET;

        return numSpikes;
    }

    // OPT: directly iterate on binary spike file; saves loading all spikes
    // a priori and slow access in multimap (brion::Spikes)
    size_t _loadSpikesFast( const float start, const float end )
    {
        const size_t size = _spikesFile->getSize();
        const size_t nElems = size / sizeof( uint32_t );
        const uint32_t* iData = _spikesFile->getAddress< uint32_t >();
        const float* fData = _spikesFile->getAddress< float >();

        if( start < _previousStart.first )
            _previousStart.second = 2; // start over, first spike after header

        size_t numSpikes = 0;
        bool once = true;
        for( size_t i = _previousStart.second; i+1 < nElems; i += 2 )
        {
            const float time = fData[ i ];
            if( time < start )
                continue;
            if( once )
            {
                // remember this time for next start that might come after
                _previousStart = std::make_pair( start, i );
                once = false;
            }

            if( time >= end )
                break;

            const uint32_t gid = iData[ i + 1 ];
            if( gid >= _gidIndex.size( ))
                continue;

            ++_spikesPerNeuron[_gidIndex[gid]];
            ++numSpikes;
        }
        return numSpikes;
    }

    // for ~5 mio spikes, this is ~200ms slower than _loadSpikesFast
    size_t _loadSpikesSlow( const float start, const float end )
    {
        size_t numSpikes = 0;
        lunchbox::ScopedWrite mutex( _getSpikesLock );
        const brain::Spikes& spikes = _spikesReader->getSpikes( start, end );
        for( const brion::Spike& spike : spikes )
        {
            if( spike.second >= _gidIndex.size( ))
                continue;

            ++_spikesPerNeuron[_gidIndex[spike.second]];
            ++numSpikes;
        }

        return numSpikes;
    }

    fivox::EventSource& _output;
    const brion::BlueConfig _config;
    const float _duration;
    float _spikesStart;
    float _spikesEnd;

    // maps GID to its index in the target
    // OPT: no (unordered)map because of constant lookup but 'wastes' memory
    // (container.size() is number of GIDs)
    brion::size_ts _gidIndex;

    // aggregates spikes for each neuron in interval
    // OPT: no (unordered)map because of constant lookup but 'wastes' memory
    // (container.size() is number of GIDs)
    brion::size_ts _spikesPerNeuron;

    // for _loadSpikesFast
    std::unique_ptr< lunchbox::MemoryMap > _spikesFile;
    std::pair< float, size_t > _previousStart;

    // for _loadSpikesSlow
    std::unique_ptr< brain::SpikeReportReader > _spikesReader;
    mutable lunchbox::Lock _getSpikesLock;
};

SpikeLoader::SpikeLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_config.getTimestep( ));
}

SpikeLoader::~SpikeLoader()
{}

Vector2f SpikeLoader::_getTimeRange() const
{
    _impl->updateTimeRange();

    // The duration of the frame needs to be considered,
    // in order to not go over the available range.
    const float spikesEnd = _impl->_spikesEnd - _impl->_duration;
    if( spikesEnd < _impl->_spikesStart )
        return Vector2f( 0.f, 0.f );

    return Vector2f( _impl->_spikesStart, spikesEnd );
}

ssize_t SpikeLoader::_load( const float time )
{
    return _impl->load( time );
}

bool SpikeLoader::_hasEnded() const
{
    return _impl->_spikesReader ? _impl->_spikesReader->hasEnded() : true;
}

}
