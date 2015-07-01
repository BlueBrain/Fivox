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
    Impl( fivox::EventSource& output, const std::string& blueconfig,
          const std::string& target, const std::string& report, const float dt )
        : _output( output )
        , _experiment( blueconfig )
        , _reader( *_experiment.reports().find( report.empty() ? "soma"
                                                               : report),
                   _experiment.cell_target( target ))
        , _currentFrameId( 0xFFFFFFFFu )
        , _dt( dt )
    {
        const bbp::Cell_Target& target_ = _experiment.cell_target( target );
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( target_, bbp::NEURONS );

        const bbp::Neurons& neurons = microcircuit.neurons();
        BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
            output.add( Event( neuron.position(), 0.f ));
    }

    bool load( const float time )
    {
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
            _output.update( i++, neuron.voltage() - brion::RESTING_VOLTAGE );

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

SomaLoader::SomaLoader( const std::string& blueconfig,
                        const std::string& target, const std::string& report,
                        const float dt )
    : _impl( new SomaLoader::Impl( *this, blueconfig, target, report, dt ))
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
