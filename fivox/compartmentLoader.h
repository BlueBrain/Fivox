// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_COMPARTMENTLOADER_H
#define FIVOX_COMPARTMENTLOADER_H

#include <fivox/defines.h>
#ifdef FIVOX_USE_BBPSDK
#  include <BBP/BBP.h>
#  include <vmmlib/util.hpp>
#  ifdef final
#    undef final
#  endif
#endif
#include <boost/shared_ptr.hpp>

namespace fivox
{
class CompartmentLoader
{
public:
    CompartmentLoader( const std::string& blueconfig,
                       const std::string& target, const float time = 0.f )
#ifdef FIVOX_USE_BBPSDK
        : _experiment( blueconfig )
        , _reader( *_experiment.reports().find( "voltage" ),
                   _experiment.cell_target( target ))
#endif
        , _min( bbp::Vector3f( std::numeric_limits< float >::max( )))
        , _max( -_min )
    {
#ifdef FIVOX_USE_BBPSDK
      const bbp::Cell_Target& target_ = _experiment.cell_target( target );
      bbp::Microcircuit& microcircuit = _experiment.microcircuit();
      microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );
      LBCHECK( loadFrame( time ));

      const bbp::Neurons& neurons = microcircuit.neurons();
      BOOST_FOREACH( const bbp::Neuron& neuron, neurons )
      {
        const bbp::Vector3f& position = neuron.soma().position();
        _min = vmml::min( _min, position );
        _max = vmml::max( _max, position );
      }
#else
      throw std::runtime_error( "Fivox compiled without BBPSDK support" );
#endif
    }

  bool loadFrame( const float time )
  {

#ifdef FIVOX_USE_BBPSDK
    if( !_reader.loadFrame( time, _frame ))
    {
      std::cerr << "Could not load frame at " << time << "ms" << std::endl;
      return false;
    }

    _experiment.microcircuit().update( _frame );
    const bbp::Neurons& neurons = _experiment.microcircuit().neurons();
    _values.clear();
    _values.reserve( neurons.size( ));

    for( bbp::Neurons::const_iterator i = neurons.begin();
         i != neurons.end(); ++i )
    {
      const bbp::Vector3f& position = i->soma().position();
      _values.push_back( bbp::Vector4f( position, i->voltage() + 65.f ));
    }
    return true;
#else
    return false;
#endif
  }

  const bbp::Vector4fs& getValues() const { return _values; }
  const bbp::Vector3f& getMin() const { return _min; }
  const bbp::Vector3f& getMax() const { return _max; }

private:
#ifdef FIVOX_USE_BBPSDK
  bbp::Experiment _experiment;
  bbp::CompartmentReportReader _reader;
  bbp::CompartmentReportFrame _frame;
#endif

  bbp::Vector4fs _values;
  bbp::Vector3f _min;
  bbp::Vector3f _max;
};

typedef boost::shared_ptr< CompartmentLoader > CompartmentLoaderPtr;
} // end namespace fivox

#endif
