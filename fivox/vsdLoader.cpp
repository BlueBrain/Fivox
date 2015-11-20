/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 *                     Juan Hernando <jhernando@fi.upm.es>
 *                     Daniel.Nachbaur@epfl.ch
 */

#include "vsdLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <BBP/BBP.h>

namespace fivox
{
class VSDLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _experiment( params.getConfig( ))
        , _target( _experiment.cell_target(
                       params.getTarget( _experiment.circuit_target( ))))
        , _voltages( *_experiment.reports().find( params.getReport( )), _target)
        , _areas( *_experiment.reports().find( "area" ), _target )
    {
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( _target, bbp::NEURONS | bbp::MORPHOLOGIES );
        _areas.updateMapping( _target );

        if( !_areas.loadFrame( 0.f, _areasFrame ))
            throw std::runtime_error( "Can't load 'area' vsd report" );

        size_t index = 0;
        for( const uint32_t gid : _target )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            output.add( Event( neuron.position(), 0.f));

            // The next statement is triggering the creation of a morphology
            // in global coordinates. We will admit it despite if effectively
            // doubles the memory usage at best because this code is to going
            // to be changed in the very near future.
            const bbp::Sections& sections = neuron.dendrites();
            for( const bbp::Section& section : sections )
            {
                const size_t nCompartments =
                    _areas.getCompartmentCounts()[index][section.id()];
                const float length = 1.f / float( nCompartments );
                for( float k = length * .5f; k < 1.0; k += length )
                    output.add(
                        Event( section.cross_section( k ).center(), 0.f ));
            }
            ++index;
        }

        const float thickness = _output.getBoundingBox().getDimension()[1];
        setCurve( fivox::AttenuationCurve( params.getDyeCurve(), thickness ));
    }

    bool load( const float time )
    {
        bbp::CompartmentReportFrame voltageFrame;
        if( !_voltages.loadFrame( time, voltageFrame ))
        {
            LBERROR << "Could not load frame at " << time << "ms" <<std::endl;
            return false;
        }

        const bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        const bbp::floatsPtr& areas = _areasFrame.getData< bbp::floatsPtr >();
        const bbp::floatsPtr& voltages =
            voltageFrame.getData< bbp::floatsPtr >();
        const float yMax = _output.getBoundingBox().getMax()[1];

        size_t eventIndex = 0;
        size_t i = 0;
        for( const uint32_t gid : _target )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            const size_t somaID = neuron.soma().id();
            assert( _voltages.getCompartmentCounts()[i][somaID] == 1 );
            // New block to avoid shadowing warnings.
            {
                // Nothing guarantees that the offsets are going to be the same.
                const uint64_t voltageOffset = _voltages.getOffsets()[i][somaID];
                const uint64_t areaOffset = _areas.getOffsets()[i][somaID];
                const float voltage = ( *voltages )[voltageOffset];
                const float area = ( *areas )[areaOffset];
                _updateEventValue( eventIndex++, voltage, area, yMax );
            }

            const bbp::Sections& sections = neuron.dendrites();
            for( const bbp::Section& section : sections )
            {
                const uint32_t id = section.id();
                const size_t nCompartments =_areas.getCompartmentCounts()[i][id];
                assert( nCompartments ==
                        _voltages.getCompartmentCounts()[i][id] );
                uint64_t voltageOffset = _voltages.getOffsets()[i][id];
                uint64_t areaOffset = _areas.getOffsets()[i][id];

                for( size_t k = 0; k < nCompartments;
                     ++k, ++eventIndex, ++voltageOffset, ++areaOffset )
                {
                    const float voltage = ( *voltages )[voltageOffset];
                    const float area = ( *areas )[areaOffset];
                    _updateEventValue( eventIndex, voltage, area, yMax );
                }
                LBVERB << section.id() << std::endl;
            }
            ++i;
        }
        return true;
    }

    void setCurve( const AttenuationCurve& curve ) { _curve = curve; }

    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    const bbp::Cell_Target _target;
    bbp::CompartmentReportReader _voltages;
    bbp::CompartmentReportReader _areas;
    bbp::CompartmentReportFrame _areasFrame;

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
        setDt( _impl->_voltages.getTimestep( ));
}

VSDLoader::~VSDLoader()
{}

void VSDLoader::setCurve( const AttenuationCurve& curve )
{
    _impl->setCurve( curve );
}

Vector2f VSDLoader::_getTimeRange() const
{
    return Vector2f( _impl->_voltages.getStartTime(),
                     _impl->_voltages.getEndTime( ));
}

bool VSDLoader::_load( const float time )
{
    return _impl->load( time );
}

}
