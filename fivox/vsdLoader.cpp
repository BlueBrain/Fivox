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
#include "uriHandler.h"

#include <brion/brion.h>
#include <brain/circuit.h>
#include <brain/morphology.h>

namespace fivox
{

class VSDLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _config( params.getConfig( ))
        , _target( _config.parseTarget( params.getTarget( )))
        , _voltageReport( _config.getReportSource( params.getReport( )),
                          brion::MODE_READ, _target)
        , _areaReport( _config.getReportSource( "area" ),
                       brion::MODE_READ, _target )
    {
        _areaReport.updateMapping( _target );
        _voltageReport.updateMapping( _target );

        const brain::Circuit circuit( _config );
        const auto morphologies = circuit.loadMorphologies(
            _target, brain::Circuit::COORDINATES_GLOBAL );

        _areas = _areaReport.loadFrame( 0.f );
        if( !_areas )
            LBTHROW( std::runtime_error( "Can't load 'area' vsd report" ));

        _circuitSectionIDs.reserve( _target.size( ));

        for( size_t i = 0; i != morphologies.size(); ++i )
        {
            const brain::Morphology& morphology = *morphologies[i];
            const auto sections =
                morphology.getSectionIDs({ brion::SECTION_SOMA,
                                           brion::SECTION_DENDRITE,
                                           brion::SECTION_APICAL_DENDRITE });

            _circuitSectionIDs.push_back( sections );

            const auto& counts = _areaReport.getCompartmentCounts()[i];

            for( auto sectionId : _circuitSectionIDs[i] )
            {
                assert( sectionId < counts.size( ));
                const size_t nCompartments = counts[sectionId];
                assert( nCompartments );

                const float length = 1.f / float( nCompartments );
                brion::floats samples;
                samples.reserve( nCompartments );
                // a sample per compartment, covering the whole section
                for( float k = length * .5f; k < 1.0; k += length )
                    samples.push_back( k );

                const auto points =
                    morphology.getSectionSamples( sectionId, samples );
                for( const auto& point : points )
                    output.add( Event( point.get_sub_vector< 3 >(), 0.f ));
            }
        }

        const float thickness = _output.getBoundingBox().getDimension()[1];
        setCurve( fivox::AttenuationCurve( params.getDyeCurve(), thickness ));
    }

    ssize_t load( const float time )
    {
        brion::floatsPtr voltages = _voltageReport.loadFrame( time );
        if( !voltages )
            return -1;

        size_t eventIndex = 0;
        const float yMax = _output.getBoundingBox().getMax()[1];
        for( size_t n = 0; n != _target.size(); ++n)
        {
            for( auto id : _circuitSectionIDs[n] )
            {
                const size_t nCompartments =
                    _areaReport.getCompartmentCounts()[n][id];
                assert( nCompartments ==
                        _voltageReport.getCompartmentCounts()[n][id] );
                uint64_t voltageOffset = _voltageReport.getOffsets()[n][id];
                uint64_t areaOffset = _areaReport.getOffsets()[n][id];

                for( size_t k = 0; k < nCompartments;
                     ++k, ++eventIndex, ++voltageOffset, ++areaOffset )
                {
                    const float voltage = ( *voltages )[voltageOffset];
                    const float area = ( *_areas )[areaOffset];
                    _updateEventValue( eventIndex, voltage, area, yMax );
                }
            }
        }
        return eventIndex;
    }

    void setCurve( const AttenuationCurve& curve ) { _curve = curve; }

    fivox::EventSource& _output;

    brion::BlueConfig _config;
    brion::GIDSet _target;
    std::vector< std::vector< uint32_t >> _circuitSectionIDs;

    brion::CompartmentReport _voltageReport;
    brion::CompartmentReport _areaReport;
    brion::floatsPtr _areas;

    AttenuationCurve _curve;

    void _updateEventValue( const size_t index, const float voltage,
                            const float area, const float yMax )
    {
        const float normVoltage = voltage - brion::MINIMUM_VOLTAGE;
        const Event& event = _output.getEvents()[index];
        const float depth = yMax - event.position[1];
        const float eventValue =
            normVoltage * area * _curve.getAttenuation( depth );
        _output.update( index, eventValue );
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
