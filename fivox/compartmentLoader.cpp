/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
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

#include "compartmentLoader.h"

#include "helpers.h"
#include "uriHandler.h"

#include <brion/brion.h>
#include <brain/circuit.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

#ifndef NDEBUG
# define DEBUG_INVERSE_MAPPING
#endif

namespace fivox
{

class CompartmentLoader::Impl
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

        helpers::addCompartmentEvents( morphologies, _report, output );
    }

    ssize_t load( const float time )
    {
        const brion::floatsPtr values = _report.loadFrame( time );
        if( !values )
            return -1;

        for( size_t i = 0; i != values->size(); ++i )
            _output[i] = ( *values )[i];

        return values->size();
    }

    EventSource& _output;
    brion::CompartmentReport _report;
};

CompartmentLoader::CompartmentLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new CompartmentLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_report.getTimestep( ));
}

CompartmentLoader::~CompartmentLoader()
{}

Vector2f CompartmentLoader::_getTimeRange() const
{
    return Vector2f( _impl->_report.getStartTime(),
                     _impl->_report.getEndTime( ));
}

ssize_t CompartmentLoader::_load( const float time )
{
    return _impl->load( time );
}

}
