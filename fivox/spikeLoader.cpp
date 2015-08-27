/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */

#include "spikeLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#include <brion/brion.h>
#include <monsteer/streaming/spikeReportReader.h>
#include <monsteer/streaming/spikes.h>
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
        , _experiment( params.getConfig( ))
        , _currentTime( -1.f )
        , _dt( params.getDt( ))
        , _duration( params.getDuration( ))
        , _spikesStart( 0.f )
        , _magnitude( 100000.f / _duration )
    {
        LBINFO << "Loading circuit..." << std::endl;
        const brion::Circuit circuit( _experiment.circuit_source() +
                                      "/circuit.mvd2" );
        const std::string& target = params.getTarget(
                                        _experiment.circuit_target( ));

        LBINFO << "Loading target " << target << "..." << std::endl;
        const brion::Targets targets{
            brion::Target( _experiment.target_source() + "/start.target" ),
            brion::Target( _experiment.user_target_source( )) };
        const brion::GIDSet& gids = brion::Target::parse( targets, target );

        if( gids.empty( ))
            LBTHROW( std::runtime_error( "No GIDs found for target '" + target +
                                         "' in " + params.getConfig( )));
        if( _dt < 0.f )
            _dt = _experiment.timestep();

        LBINFO << "Loading spikes for " << gids.size() << " cells..."
               << std::endl;
        const brion::NeuronMatrix& matrix =
            circuit.get( gids, brion::NEURON_POSITION_X |
                               brion::NEURON_POSITION_Y |
                               brion::NEURON_POSITION_Z );

        size_t i = 0;
        _gidIndex.resize( *gids.rbegin() + 1 );
        for( const uint32_t gid: gids )
        {
            const Vector3f position( lexical_cast< float >( matrix[i][0] ),
                                     lexical_cast< float >( matrix[i][1] ),
                                     lexical_cast< float >( matrix[i][2] ));
            _output.add( Event( position, 0.f ));
            _gidIndex[gid] = i++;
        }
        _spikesPerNeuron.resize( gids.size( ));
        _loadSpikes( params.getSpikes( ));

        LBINFO << "Finished loading, magnitude " << _magnitude << ", "
               << gids.size() << " neurons" << std::endl;
    }

    void _loadSpikes( std::string spikes )
    {
        if( spikes.empty( ))
            spikes = _experiment.spikes_source().getPath();

        if( _loadBinarySpikes( spikes ))
            return;

        LBINFO << "No valid binary .spikes file found, loading from .dat..."
               << std::endl;
        _spikesReader.reset(
                    new monsteer::SpikeReportReader( lunchbox::URI( spikes )));
        _spikesStart = _spikesReader->isStream() ? 0.0 :
                                                _spikesReader->getStartTime();
        _spikesEnd = _spikesReader->isStream() ? 0.0 :
                                                _spikesReader->getEndTime();
    }

    bool _loadBinarySpikes( const std::string& spikes )
    {
        std::unique_ptr< lunchbox::MemoryMap > spikesFile
                ( new lunchbox::MemoryMap( spikes ));
        const size_t size = spikesFile->getSize();
        if( (size % sizeof( uint32_t )) != 0 )
            return false;
        const size_t nElems = spikesFile->getSize() / sizeof( uint32_t );
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
        _spikesStart = fData[index]; // first spike after header
        return true;
    }

    bool load( const float start )
    {
        if( start == _currentTime )
            return false;
        _currentTime = start;

        lunchbox::setZero( _spikesPerNeuron.data(),
                           _spikesPerNeuron.size() * sizeof(size_t));

        const float end = start + _duration;
        const size_t numSpikes = _spikesFile ? _loadSpikesFast( start, end )
                                             : _loadSpikesSlow( start, end );

        for( size_t i = 0; i < _spikesPerNeuron.size(); ++i )
            _output.update( i, _spikesPerNeuron[i] * _magnitude );

        LBINFO << "Loaded " << numSpikes << " spikes from " << start << " to "
               << end << " ms" << std::endl;

        return true;
    }

    bool load( const uint32_t frame )
    {
        if( !_output.isInFrameRange( frame ))
            return false;

        const float time = _dt * frame;
        return load( time );
    }

    Vector2ui getFrameRange( )
    {
        if( _spikesReader->isStream( ) )
        {
            lunchbox::ScopedWrite mutex( _getSpikesLock );
            const monsteer::Spikes& spikes = _spikesReader->getSpikes();
            if( !spikes.empty( ))
            {
                _spikesStart = spikes.getStartTime();
                _spikesEnd = spikes.getEndTime();
            }
        }
        return Vector2ui( _spikesStart / _dt,
                          _spikesEnd / _dt );
    }

private:
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
            else if( once )
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
        const monsteer::Spikes& spikes = _spikesReader->getSpikes( start, end );
        for( const bbp::Spike& spike: spikes )
        {
            if( spike.second >= _gidIndex.size( ))
                continue;

            ++_spikesPerNeuron[_gidIndex[spike.second]];
            ++numSpikes;
        }

        return numSpikes;
    }

    fivox::EventSource& _output;
    const bbp::Experiment_Specification _experiment;
    float _currentTime;
    float _dt;
    const float _duration;
    float _spikesStart;
    float _spikesEnd;
    const float _magnitude;

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
    std::unique_ptr< monsteer::SpikeReportReader > _spikesReader;
    mutable lunchbox::Lock _getSpikesLock;
};

SpikeLoader::SpikeLoader( const URIHandler& params )
    : _impl( new Impl( *this, params ))
{}

SpikeLoader::~SpikeLoader()
{}

bool SpikeLoader::load( const float time )
{
    return _impl->load( time );
}

bool SpikeLoader::load( const uint32_t frame )
{
    return _impl->load( frame );
}

Vector2ui SpikeLoader::getFrameRange()
{
    return _impl->getFrameRange();
}

}
