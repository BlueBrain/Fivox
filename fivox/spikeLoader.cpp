/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "spikeLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#include <brion/brion.h>
#include <lunchbox/bitOperation.h>
#ifdef final
#  undef final
#endif

using boost::lexical_cast;

namespace fivox
{
namespace detail
{

class SpikeLoader
{
public:
    SpikeLoader( fivox::EventSource& output,
                 const bbp::Experiment_Specification& spec,
                 const std::string& spikeFile, const float time,
                 const float window )
        : _output( output )
        , _spikes( spikeFile.empty() ? spec.spikes_source() :
                                       lunchbox::URI( spikeFile ),
                   brion::MODE_READ )
        , _circuit( spec.circuit_source() + "/circuit.mvd2" )
    {
        // compute bounding box to set correct size for partial spikes
        const brion::Spikes& spikes = _spikes.getSpikes();
        brion::GIDSet gids;
        BOOST_FOREACH( const brion::Spike& spike, spikes )
            gids.insert( spike.second );

        const brion::NeuronMatrix& matrix =
            _circuit.get( gids, brion::NEURON_POSITION_X |
                                brion::NEURON_POSITION_Y |
                                brion::NEURON_POSITION_Z );
        for( size_t i = 0; i < gids.size(); ++i )
        {
            _bbox.merge( Vector3f( lexical_cast< float >( matrix[i][0] ),
                                   lexical_cast< float >( matrix[i][1] ),
                                   lexical_cast< float >( matrix[i][2] )));
        }

        // Load requested data
        LBCHECK( loadFrame( time, window ));
    }

    bool loadFrame( const float start, const float window )
    {
        _output.clear();

        const float end = start + window;
        _spikes.waitUntil( end );

        const brion::Spikes& spikes = _spikes.getSpikes();
        brion::GIDSet gids;
        BOOST_FOREACH( const brion::Spike& spike, spikes )
            if( start <= spike.first && end > spike.first )
                gids.insert( spike.second );

        LBINFO << gids.size() << " spikes in " << start << ", " << end
               << std::endl;
        const brion::NeuronMatrix& matrix =
            _circuit.get( gids, brion::NEURON_POSITION_X |
                                brion::NEURON_POSITION_Y |
                                brion::NEURON_POSITION_Z );
        size_t index = 0;
        BOOST_FOREACH( uint32_t gid, gids )
        {
            const Vector3f pos( lexical_cast< float >( matrix[index][0] ),
                                lexical_cast< float >( matrix[index][1] ),
                                lexical_cast< float >( matrix[index][2] ));
            _output.add( Event( pos, 20.f ));
            ++index;
            LBVERB << gid << " at " << pos << std::endl;
        }

        // Add empty corner points to set size correctly
        _output.add( Event( _bbox.getMin(), 0.f ));
        _output.add( Event( _bbox.getMax(), 0.f ));
        return true;
    }

private:
    fivox::EventSource& _output;
    brion::SpikeReport _spikes;
    brion::Circuit _circuit;
    vmml::AABBf _bbox;
};
}

namespace
{
};

SpikeLoader::SpikeLoader( const bbp::Experiment_Specification& spec,
                          const std::string& spikes, const float time,
                          const float window )
    : _impl( new detail::SpikeLoader( *this, spec, spikes, time, window ))
{}

SpikeLoader::~SpikeLoader()
{
    delete _impl;
}

bool SpikeLoader::loadFrame( const float time, const float window )
{
    return _impl->loadFrame( time, window );
}

}
