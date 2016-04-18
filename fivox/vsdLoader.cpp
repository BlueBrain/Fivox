/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
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

#include "vsdLoader.h"

#include "event.h"
#include "helpers.h"
#include "uriHandler.h"

#include <brion/brion.h>
#include <brain/circuit.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

#include <cassert>

namespace fivox
{

class VSDLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _config( params.getConfig( ))
        , _target( _config.parseTarget( params.getTarget(
                                            _config.getCircuitTarget( ))))
        , _voltageReport( _config.getReportSource( params.getReport( )),
                          brion::MODE_READ, _target)
        , _areaReport( _config.getReportSource( "areas" ),
                       brion::MODE_READ, _target )
    {
        _areaReport.updateMapping( _target );
        _voltageReport.updateMapping( _target );

        const brain::Circuit circuit( _config );
        const auto morphologies = circuit.loadMorphologies(
            _target, brain::Circuit::COORDINATES_GLOBAL );

        _areas = _areaReport.loadFrame( 0.f );
        if( !_areas )
            LBTHROW( std::runtime_error( "Can't load 'areas' vsd report" ));

        helpers::addCompartmentEvents( morphologies, _voltageReport, output );

        const float thickness = _output.getBoundingBox().getSize()[1];
        setCurve( fivox::AttenuationCurve( params.getDyeCurve(), thickness ));
    }

    ssize_t load( const float time )
    {
        brion::floatsPtr voltages = _voltageReport.loadFrame( time );
        if( !voltages )
            return -1;

        const float yMax = _output.getBoundingBox().getMax()[1];

        assert( voltages->size() == _areas->size( ));
        for( size_t i = 0; i != voltages->size( ); ++i )
            _updateEventValue( i, ( *voltages )[i], ( *_areas )[i], yMax );

        return voltages->size();
    }

    void setCurve( const AttenuationCurve& curve ) { _curve = curve; }

    fivox::EventSource& _output;

    brion::BlueConfig _config;
    brion::GIDSet _target;

    brion::CompartmentReport _voltageReport;
    brion::CompartmentReport _areaReport;
    brion::floatsPtr _areas;

    AttenuationCurve _curve;

    void _updateEventValue( const size_t index, const float voltage,
                            const float area, const float yMax )
    {
        const Event& event = _output.getEvents()[index];
        const float depth = yMax - event.position[1];
        const float eventValue = voltage * area *
                                 _curve.getAttenuation( depth );
        _output[index].value = eventValue;
    }
};

VSDLoader::VSDLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new VSDLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_voltageReport.getTimestep( ));
}

VSDLoader::~VSDLoader()
{}

void VSDLoader::setCurve( const AttenuationCurve& curve )
{
    _impl->setCurve( curve );
}

Vector2f VSDLoader::_getTimeRange() const
{
    return Vector2f( _impl->_voltageReport.getStartTime(),
                     _impl->_voltageReport.getEndTime( ));
}

ssize_t VSDLoader::_load( const float time )
{
    return _impl->load( time );
}

}
