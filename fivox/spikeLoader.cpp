/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "spikeLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#include <brion/brion.h>
#include <lunchbox/bitOperation.h>
#include <lunchbox/stdExt.h>
#include <boost/foreach.hpp>
#include <boost/unordered_map.hpp>
#ifdef final
#  undef final
#endif

using boost::lexical_cast;
using brion::Strings;

namespace fivox
{
namespace detail
{

void _loadTarget( brion::GIDSet& gids, const brion::Target& target,
                  const std::string& name )
{
    const Strings& values = target.get( name );
    BOOST_FOREACH( const std::string& value, values )
    {
        try
        {
            gids.insert( lexical_cast< uint32_t >( value.substr( 1 )));
        }
        catch( ... )
        {
            if( value != name )
                _loadTarget( gids, target, value );
        }
    }
}

class SpikeLoader
{
public:
    SpikeLoader( fivox::EventSource& output,
                 const std::string& blueconfig,
                 const std::string& spikeFile, const float time,
                 const float duration )
        : _output( output )
        , _experiment( blueconfig )
        , _spikes( spikeFile.empty() ? _experiment.spikes_source() :
                                       lunchbox::URI( spikeFile ))
        , _magnitude( .5f )
    {
        // Get all neuron positions and compute bounding box to set correct size
        // for partial spikes
        const brion::Circuit circuit( _experiment.circuit_source() +
                                      "/circuit.mvd2" );
        const brion::Target target( _experiment.target_source() +
                                    "/start.target" );
        brion::GIDSet gids;
        _loadTarget( gids, target, _experiment.circuit_target( ));
        _magnitude = 100.f / std::log( gids.size( )); // heuristic

        const brion::NeuronMatrix& matrix =
            circuit.get( gids, brion::NEURON_POSITION_X |
                               brion::NEURON_POSITION_Y |
                               brion::NEURON_POSITION_Z );

        size_t i = 0;
        BOOST_FOREACH( uint32_t gid, gids )
        {
            const Vector3f position( lexical_cast< float >( matrix[i][0] ),
                                     lexical_cast< float >( matrix[i][1] ),
                                     lexical_cast< float >( matrix[i][2] ));
            _positions[ gid ] = position;
            _bbox.merge( position );
            ++i;
        }

        // Load requested data
        LBCHECK( loadFrame( time, duration ));
    }

    bool loadFrame( const float start, const float duration )
    {
        _output.clear();

        const bbp::Spikes& spikes = _spikes.getSpikes( start, start+duration );

        BOOST_FOREACH( const bbp::Spike& spike, spikes )
            _output.add( Event( _positions[ spike.second ], _magnitude ));

        LBINFO << spikes.size() << " spikes of magnitude " << _magnitude
               << " in " << duration << " ms after " << start << " in "
               << _bbox << std::endl;

        // Add empty corner points to set size correctly
        _output.add( Event( _bbox.getMin(), 0.f ));
        _output.add( Event( _bbox.getMax(), 0.f ));
        return true;
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment_Specification _experiment;
    bbp::SpikeReportReader _spikes;
    vmml::AABBf _bbox;
    float _magnitude;

    typedef boost::unordered_map< uint32_t, Vector3f > Positions;
    Positions _positions;
};
}

namespace
{
};

SpikeLoader::SpikeLoader( const std::string& blueconfig,
                          const std::string& spikes, const float time,
                          const float duration )
    : _impl( new detail::SpikeLoader( *this, blueconfig, spikes, time,
                                      duration ))
{}

SpikeLoader::~SpikeLoader()
{
    delete _impl;
}

bool SpikeLoader::loadFrame( const float time, const float duration )
{
    return _impl->loadFrame( time, duration );
}

}
