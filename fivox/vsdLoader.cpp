/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */

#include "vsdLoader.h"
#include "event.h"

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
        , _currentFrameId( 0xFFFFFFFFu )
        , _dt( params.getDt( ))
    {
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( _target, bbp::NEURONS | bbp::MORPHOLOGIES );
        _areas.updateMapping( _target );

        if( !_areas.loadFrame( 0.f, _areasFrame ))
            throw std::runtime_error( "Can't load 'area' vsd report" );

        if( _dt < 0.f )
            _dt = _voltages.getTimestep();

        size_t i = 0;
        for( const uint32_t gid : _target )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            const bbp::Sections& sections = neuron.dendrites();
            size_t j = 0;
            for( const bbp::Section& section : sections )
            {
                const size_t nCompartments =_areas.getCompartmentCounts()[i][j];
                const float compartmentLength = 1.f / float( nCompartments );
                const float compartmentMiddle = compartmentLength * .5f;

                for( size_t k = 0; k < nCompartments; ++k )
                {
                    const bbp::Cross_Section& midPoint = section.cross_section(
                        compartmentMiddle + k * compartmentLength );

                    output.add( Event( midPoint.center(), 0.f ));
                }
                ++j;
            }
            ++i;
        }

        const float thickness = _output.getBoundingBox().getDimension()[1];
        setCurve( fivox::AttenuationCurve( params.getDyeCurve(), thickness ));
    }

    bool load( const float time )
    {
        bbp::CompartmentReportFrame voltages;
        if( !_voltages.loadFrame( time, voltages ))
        {
            LBERROR << "Could not load frame at " << time << "ms" <<std::endl;
            return false;
        }

        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.update( voltages );
        const bbp::floatsPtr& areas = _areasFrame.getData< bbp::floatsPtr >();

        size_t i=0;
        size_t index = 0;
        const float yMax = _output.getBoundingBox().getMax()[1];
        for( const uint32_t gid : _target )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            const bbp::Sections& sections = neuron.dendrites();

            size_t j = 0;
            for( const bbp::Section& section : sections )
            {
                const size_t nCompartments =_areas.getCompartmentCounts()[i][j];
                uint64_t offset = _areas.getOffsets()[i][j];

                for( size_t k = 0; k < nCompartments; ++k )
                {
                    const float normVoltage = neuron.voltage() -
                                              brion::MINIMUM_VOLTAGE;

                    const Event& event = _output.getEvents()[index];
                    const float depth = yMax - event.position[1];
                    const float eventValue = normVoltage * (*areas)[offset++] *
                                                 _curve.getAttenuation( depth );
                    _output.update( index++, eventValue );
                }
                ++j;
                LBVERB << section.id() << std::endl;
            }
            ++i;
        }
        return true;
    }

    bool load( const uint32_t frame )
    {
        if( frame == _currentFrameId || !_output.isInFrameRange( frame ))
            return false;

        _currentFrameId = frame;
        const float time  = _voltages.getStartTime() + _dt * frame;
        return load( time );
    }

    void setCurve( const AttenuationCurve& curve ) { _curve = curve; }

    Vector2ui getFrameRange()
    {
        return Vector2ui( std::floor( _voltages.getStartTime() / _dt ),
                          std::ceil( _voltages.getEndTime() / _dt ));
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    const bbp::Cell_Target _target;
    bbp::CompartmentReportReader _voltages;
    bbp::CompartmentReportReader _areas;
    bbp::CompartmentReportFrame _areasFrame;

    AttenuationCurve _curve;

    uint32_t _currentFrameId;
    float _dt;
};

VSDLoader::VSDLoader( const URIHandler& params )
    : _impl( new VSDLoader::Impl( *this, params ))
{}

VSDLoader::~VSDLoader()
{}

bool VSDLoader::load( const float time )
{
    return _impl->load( time );
}

bool VSDLoader::load( const uint32_t frame )
{
    return _impl->load( frame );
}

void VSDLoader::setCurve( const AttenuationCurve& curve )
{
    _impl->setCurve( curve );
}

Vector2ui VSDLoader::getFrameRange()
{
    return _impl->getFrameRange();
}


}
