/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "synapseLoader.h"
#include "event.h"

#include <BBP/BBP.h>
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
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _experiment( params.getConfig( ))
    {
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

        LBINFO << "Loading synapses for " << gids.size() << " cells..."
               << std::endl;
        boost::progress_display progress( gids.size( ));
        const brion::Synapse synapses( _experiment.synapse_source() +
                                       "/nrn_positions.h5" );

        for( const uint32_t gid : gids )
        {
            const brion::SynapseMatrix& data =
                synapses.read( gid, brion::SYNAPSE_PRESYNAPTIC_SURFACE_X |
                                    brion::SYNAPSE_PRESYNAPTIC_SURFACE_Y |
                                    brion::SYNAPSE_PRESYNAPTIC_SURFACE_Z );
            for( size_t i = 0; i < data.shape()[0]; ++i )
                _output.add( Event( Vector3f( data[i][0], data[i][1],
                                              data[i][2] ), 0.f ));
            ++progress;
        }

        const Events& events = _output.getEvents();
        const float magnitude = 15000.f / events.size(); // heuristic
        for( size_t i = 0; i < events.size(); ++i )
            _output.update( i, magnitude );
    }

private:
    fivox::EventSource& _output;
    const bbp::Experiment_Specification _experiment;
};

SynapseLoader::SynapseLoader( const URIHandler& params )
    : _impl( new Impl( *this, params ))
{}

SynapseLoader::~SynapseLoader()
{}

}
