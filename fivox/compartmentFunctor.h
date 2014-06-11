// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_COMPARTMENTFUNCTOR_H
#define FIVOX_COMPARTMENTFUNCTOR_H

#include <fivox/defines.h>
#ifdef FIVOX_USE_BBPSDK
#  include <BBP/BBP.h>
#  ifdef final
#    undef final
#  endif
#endif
#include <itkNumericTraits.h>
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
      microcircuit.load( target_, bbp::NEURONS );
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
    return true;
  }

  const bbp::Neurons& getNeurons() const
      { return _experiment.microcircuit().neurons(); }
#endif

private:
#ifdef FIVOX_USE_BBPSDK
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;
    bbp::CompartmentReportFrame _frame;
#endif
};

typedef boost::shared_ptr< CompartmentLoader > CompartmentLoaderPtr;

/** BBPSDK based functor sampling compartments into the given pixel. */
template< typename TImage > class CompartmentFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;

    CompartmentFunctor() {}
    ~CompartmentFunctor() {}

    bool operator!=(const CompartmentFunctor &) const { return false; }
    bool operator==(const CompartmentFunctor & other) const
        { return !( *this != other ); }


    inline TPixel operator()( const TPoint& /*point*/ ) const
    {
      if( !_loader )
        return 0;

      TAccumulator sum = 0;
#ifdef FIVOX_USE_BBPSDK
      const bbp::Neurons& neurons = _loader->getNeurons();
      for( bbp::Neurons::const_iterator i = neurons.begin();
           i != neurons.end(); ++i )
      {
        sum += i->voltage();
      }
#endif
      return sum;
    }

    void setLoader( CompartmentLoaderPtr loader ) { _loader = loader; }

private:
    CompartmentLoaderPtr _loader;
};
} // end namespace fivox

#endif
