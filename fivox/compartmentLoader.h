// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_COMPARTMENTLOADER_H
#define FIVOX_COMPARTMENTLOADER_H

#include <fivox/defines.h>
#ifdef FIVOX_USE_BBPSDK
#  include <BBP/BBP.h>
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
    {
      const bbp::Cell_Target& target_ = _experiment.cell_target( target );
      bbp::Microcircuit& microcircuit = _experiment.microcircuit();
      microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );
      loadFrame( time );
    }
#else
    {
        throw std::runtime_error( "Fivox compiled without BBPSDK support" );
    }
#endif

#ifdef FIVOX_USE_BBPSDK
  bool loadFrame( const float time )
  {
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
      _values.push_back( bbp::Vector4f( i->soma().position(),
                                        i->voltage() + 65.f /* rest pot. */));
    }
    return true;
  }

  const bbp::Vector4fs& getValues() const { return _values; }
#endif

private:
#ifdef FIVOX_USE_BBPSDK
  bbp::Experiment _experiment;
  bbp::CompartmentReportReader _reader;
  bbp::CompartmentReportFrame _frame;
  bbp::Vector4fs _values;
#endif
};

typedef boost::shared_ptr< CompartmentLoader > CompartmentLoaderPtr;
} // end namespace fivox

#endif
