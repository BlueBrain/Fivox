// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "somaLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#ifdef final
#  undef final
#endif

namespace fivox
{
namespace detail
{
class SomaLoader
{
public:
  SomaLoader( fivox::EventSource& output, const std::string& blueconfig,
              const std::string& target, const float time )
      : _output( output )
      , _experiment( blueconfig )
      , _reader( *_experiment.reports().find( "voltage" ),
                 _experiment.cell_target( target ))
  {
    const bbp::Cell_Target& target_ = _experiment.cell_target( target );
    bbp::Microcircuit& microcircuit = _experiment.microcircuit();
    microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );

    const bbp::Neurons& neurons = microcircuit.neurons();
    BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
      output.add( Event( neuron.soma().position(), 0.f ));

    LBCHECK( loadFrame( time ));
  }

  bool loadFrame( const float time )
  {
    bbp::CompartmentReportFrame frame;
    if( !_reader.loadFrame( time, frame ))
    {
      std::cerr << "Could not load frame at " << time << "ms" << std::endl;
      return false;
    }

    _experiment.microcircuit().update( frame );
    const bbp::Neurons& neurons = _experiment.microcircuit().neurons();
    size_t i = 0;
    BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
      _output.update( i++, neuron.voltage() + 65.f );

    return true;
  }

private:
  fivox::EventSource& _output;
  bbp::Experiment _experiment;
  bbp::CompartmentReportReader _reader;
};
}

SomaLoader::SomaLoader( const std::string& blueconfig,
                        const std::string& target, const float time )
    : _impl( new detail::SomaLoader( *this, blueconfig, target, time ))
{}

SomaLoader::~SomaLoader()
{
  delete _impl;
}

bool SomaLoader::loadFrame( const float time )
{
  return _impl->loadFrame( time );
}

}
