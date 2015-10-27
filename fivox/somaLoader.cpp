/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 */

#include "somaLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#include <boost/foreach.hpp>
#ifdef final
#  undef final
#endif

namespace fivox
{
class SomaLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _experiment( params.getConfig( ))
        , _reader( *_experiment.reports().find( params.getReport( )),
                   _experiment.cell_target(
                       params.getTarget( _experiment.circuit_target( ))))
        , _currentTime( -1.f )
        , _dt( params.getDt( ))
        , _magnitude( params.getMagnitude( ))
    {
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( _reader.getCellTarget(), bbp::NEURONS );

        const bbp::Neurons& neurons = microcircuit.neurons();
        BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
            output.add( Event( neuron.position(), 0.f ));
    }

    bool load( const float time )
    {
        if( time == _currentTime )
            return true;
        _currentTime = time;

        bbp::CompartmentReportFrame frame;
        if( !_reader.loadFrame( time, frame ))
        {
            LBERROR << "Could not load frame at " << time << "ms" << std::endl;
            return false;
        }

        if( _dt < 0.f )
            _dt = _reader.getTimestep();

        _experiment.microcircuit().update( frame );
        const bbp::Neurons& neurons = _experiment.microcircuit().neurons();
        size_t i = 0;
        BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
            _output.update( i++, _magnitude * ( neuron.voltage() -
                                                brion::MINIMUM_VOLTAGE ));
        return true;
    }

    bool load( const uint32_t frame )
    {
        if( !_output.isInFrameRange( frame ))
            return false;

        const float time  = _reader.getStartTime() + _dt * frame;
        return load( time );
    }

    Vector2ui getFrameRange()
    {
        return Vector2ui( std::floor( _reader.getStartTime() / _dt ),
                          std::ceil( _reader.getEndTime() / _dt ));
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;

    float _currentTime;
    float _dt;
    const float _magnitude;
};

SomaLoader::SomaLoader( const URIHandler& params )
    : _impl( new SomaLoader::Impl( *this, params ))
{}

SomaLoader::~SomaLoader()
{}

bool SomaLoader::load( const float time )
{
    return _impl->load( time );
}

bool SomaLoader::load( const uint32_t frame )
{
    return _impl->load( frame );
}

Vector2ui SomaLoader::getFrameRange()
{
    return _impl->getFrameRange();
}

}
