/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
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
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _config( params.getConfig( ))
    {
        const auto gids = _config.parseTarget(
                                params.getTarget( _config.getCircuitTarget( )));

        if( gids.empty( ))
        {
            LBTHROW( std::runtime_error(
                         "No GIDs found for requested target in " +
                         params.getConfig( )));
        }

        LBINFO << "Loading synapses for " << gids.size() << " cells..."
               << std::endl;
        boost::progress_display progress( gids.size( ));
        const brion::Synapse synapses( _config.getSynapseSource().getPath() +
                                       "/nrn_positions.h5" );
        const float magnitude = params.getMagnitude();

        for( const uint32_t gid : gids )
        {
            const brion::SynapseMatrix& data =
                synapses.read( gid, brion::SYNAPSE_PRESYNAPTIC_SURFACE_X |
                                    brion::SYNAPSE_PRESYNAPTIC_SURFACE_Y |
                                    brion::SYNAPSE_PRESYNAPTIC_SURFACE_Z );
            for( size_t i = 0; i < data.shape()[0]; ++i )
                _output.add( Event( Vector3f( data[i][0], data[i][1],
                                              data[i][2] ), magnitude ));
            ++progress;
        }
    }

private:
    fivox::EventSource& _output;
    brion::BlueConfig _config;
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
