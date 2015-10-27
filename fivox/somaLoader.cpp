/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 */

#include "somaLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <brion/brion.h>
#include <brain/circuit.h>
#include <lunchbox/bitOperation.h>

#ifdef final
#  undef final
#endif

namespace fivox
{
using boost::lexical_cast;

class SomaLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _config( params.getConfig( ))
        , _report( _config.getReportSource( params.getReport( )),
                   brion::MODE_READ, _config.parseTarget( params.getTarget( )))
    {
        const brion::GIDSet& gids = _report.getGIDs();
        brain::Circuit circuit( _config );
        brain::Vector3fs positions = circuit.getPositions( gids );

        for( size_t i = 0; i < gids.size(); ++i )
            output.add( Event( positions[i], 0.f ));
    }

    bool load( const float time )
    {
        const brion::floatsPtr frame = _report.loadFrame( time );
        if( !frame )
        {
            LBERROR << "Could not load frame at " << time << "ms" << std::endl;
            return false;
        }

        const brion::GIDSet& gids = _report.getGIDs();
        const brion::SectionOffsets& offsets = _report.getOffsets();
        const std::vector< float > voltages = *frame;

        for( size_t i = 0; i < gids.size(); ++i )
        {
            // This code assumes that section 0 is the soma.
            const float v = voltages[offsets[i][0]] - brion::MINIMUM_VOLTAGE;
            _output.update( i, v );
        }
        return true;
    }

    fivox::EventSource& _output;
    brion::BlueConfig _config;
    brion::CompartmentReport _report;
};

SomaLoader::SomaLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new SomaLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_report.getTimestep( ));
}

SomaLoader::~SomaLoader()
{}

Vector2f SomaLoader::_getTimeRange() const
{
    return Vector2f( _impl->_report.getStartTime(),
                     _impl->_report.getEndTime( ));
}

bool SomaLoader::_load( const float time )
{
    return _impl->load( time );
}

}
