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
        , _currentFrameId( 0xFFFFFFFFu )
        , _dt( params.getDt( ))
    {
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( _reader.getCellTarget(), bbp::NEURONS );

        const bbp::Neurons& neurons = microcircuit.neurons();
        BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
            output.add( Event( neuron.position(), 0.f ));
    }

    bool load( const float time )
    {
        static const float magnitude = .1f; // heuristic
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
            _output.update( i++, magnitude * ( neuron.voltage() -
                                               brion::RESTING_VOLTAGE ));
        return true;
    }

    void load( const uint32_t frame )
    {
        if( frame == _currentFrameId )
            return;

        _currentFrameId = frame;
        const float time  = _reader.getStartTime() + _dt * frame;
        LBCHECK( load( time ));
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;

    uint32_t _currentFrameId;
    float _dt;
};

SomaLoader::SomaLoader( const URIHandler& params )
    : _impl( new SomaLoader::Impl( *this, params ))
{}

SomaLoader::~SomaLoader()
{}

void SomaLoader::load( const float time )
{
    _impl->load( time );
}

void SomaLoader::load( const uint32_t frame )
{
    _impl->load( frame );
}

}
