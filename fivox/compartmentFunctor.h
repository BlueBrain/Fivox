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
      _values.push_back( bbp::Vector4f( i->soma().position(), i->voltage( )));
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


    inline TPixel operator()( const TPoint& point ) const
    {
      if( !_loader )
        return 0;

      TAccumulator sum = 0;
#ifdef FIVOX_USE_BBPSDK
      static const float rho = 3.54f; //omh*m == 354 ohm*cm
      static const float factor = rho /( 4.f * M_PI );
      static const float threshold = 5.f;

      bbp::Vector3f base;
      const size_t components = std::min( point.Size(), 3u );
      for( size_t i = 0; i < components; ++i )
        base[i] = point[i];

      const bbp::Vector4fs& values = _loader->getValues();
      for( bbp::Vector4fs::const_iterator i = values.begin();
           i != values.end(); ++i )
      {
		const bbp::Vector3f& position = i->get_sub_vector<3>();
        const float distance = (base - position).length();
        if( distance < threshold )
          continue;

        if( distance < 1.f )
          sum += (*i)[3] * factor;
        else
          sum += (*i)[3] * factor / distance;
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
