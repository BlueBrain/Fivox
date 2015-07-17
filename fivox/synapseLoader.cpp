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
    Impl( fivox::EventSource& output, const std::string& blueconfig,
          std::string target )
        : _output( output )
        , _experiment( blueconfig )
    {
        LBINFO << "Loading target " << target << "..." << std::endl;
        if( target.empty( ))
            target = _experiment.circuit_target();
        const brion::Targets targets{
            brion::Target( _experiment.target_source() + "/start.target" ),
            brion::Target( _experiment.user_target_source( )) };
        const brion::GIDSet& gids = brion::Target::parse( targets, target );

        if( gids.empty( ))
            LBTHROW( std::runtime_error( "No GIDs found for target '" + target +
                                         "' in " + blueconfig ));

        const float magnitude = 1.f / std::log( gids.size( )); // heuristic

        LBINFO << "Loading synapses for " << gids.size() << " cells..."
               << std::endl;
        boost::progress_display progress( gids.size( ));
        const brion::Synapse synapses( _experiment.synapse_source() +
                                       "/nrn_positions.h5" );

        for( const uint32_t gid : gids )
        {
            const brion::SynapseMatrix& data =
                synapses.read( gid, brion::SYNAPSE_POSITION );
            for( size_t i = 0; i < data.shape()[0]; ++i )
                _output.add( Event( Vector3f( data[i][1], data[i][2],
                                              data[i][3] ), magnitude ));
            ++progress;
        }
    }

private:
    fivox::EventSource& _output;
    const bbp::Experiment_Specification _experiment;
};

SynapseLoader::SynapseLoader( const std::string& blueconfig,
                              const std::string& target )
    : _impl( new Impl( *this, blueconfig, target ))
{
}

SynapseLoader::~SynapseLoader()
{}

}
