/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
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

#include "somaLoader.h"
#include "event.h"
#include "helpers.h"
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
    Impl( EventSource& output, const URIHandler& params )
        : _output( output )
        , _report( params.getConfig().getReportSource( params.getReport( )),
                   brion::MODE_READ, params.getGIDs( ))
    {
        const brain::Circuit circuit( params.getConfig( ));
        const auto morphologies = circuit.loadMorphologies(
            params.getGIDs(), brain::Circuit::COORDINATES_GLOBAL );

        // add soma events only
        helpers::addCompartmentEvents( morphologies, _report, output, true );
    }

    ssize_t load( const float time )
    {
        const brion::floatsPtr frame = _report.loadFrame( time );
        if( !frame )
            return -1;

        const brion::GIDSet& gids = _report.getGIDs();
        const brion::SectionOffsets& offsets = _report.getOffsets();
        const std::vector< float > reportValues = *frame;

        for( size_t i = 0; i < gids.size(); ++i )
        {
            // This code assumes that section 0 is the soma.
            const float v = reportValues[offsets[i][0]];
            _output[i] = v;
        }
        return gids.size();
    }

    EventSource& _output;
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

ssize_t SomaLoader::_load( const float time )
{
    return _impl->load( time );
}

}
