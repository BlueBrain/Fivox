// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_COMPARTMENTFUNCTOR_H
#define FIVOX_COMPARTMENTFUNCTOR_H

#include <fivox/defines.h>
#include <fivox/compartmentLoader.h>
#include <itkNumericTraits.h>
#include <boost/shared_ptr.hpp>

namespace fivox
{
namespace
{
template< class T > inline T _scale( const float value )
    { return value; }
template<> inline unsigned char _scale( const float value )
    { return value * 256.f; }
}

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
    static const float threshold2 = 10.f * 10.f;

    bbp::Vector3f base;
    const size_t components = std::min( point.Size(), 3u );
    for( size_t i = 0; i < components; ++i )
      base[i] = point[i];

    const bbp::Vector4fs& values = _loader->getValues();
    for( bbp::Vector4fs::const_iterator i = values.begin();
         i != values.end(); ++i )
    {
      const bbp::Vector3f& position = i->get_sub_vector<3>();
      const float distance2 = (base - position).squared_length();
      if( distance2 < threshold2 )
        continue;

      if( distance2 < 1.f )
        sum += _scale< TPixel >( (*i)[3] * factor );
      else
        sum += _scale< TPixel >( (*i)[3] * factor / std::sqrt( distance2 ));
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
