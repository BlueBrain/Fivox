// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENTFUNCTOR_H
#define FIVOX_EVENTFUNCTOR_H

#include <fivox/defines.h>
#include <fivox/eventSource.h>

#pragma GCC diagnostic ignored "-Wtype-limits"
#include <itkNumericTraits.h>

#include <boost/foreach.hpp>
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

/** Functor sampling spatial events into the given pixel. */
template< typename TImage > class EventFunctor
{
public:
  typedef typename TImage::PixelType TPixel;
  typedef typename TImage::PointType TPoint;
  typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;

  EventFunctor() {}
  ~EventFunctor() {}

  bool operator!=(const EventFunctor &) const { return false; }
  bool operator==(const EventFunctor & other) const
    { return !( *this != other ); }


  inline TPixel operator()( const TPoint& point ) const
  {
    if( !_source )
      return 0;

    TAccumulator sum = 0;
    static const float rho = 3.54f; //omh*m == 354 ohm*cm
    static const float factor = rho /( 4.f * M_PI ) / 10.f;
    static const float threshold2 = 1000.f;

    Vector3f base;
    const size_t components = std::min( point.Size(), 3u );
    for( size_t i = 0; i < components; ++i )
      base[i] = point[i];

    const Events& events = _source->getEvents();
    BOOST_FOREACH( const Event& event, events )
    {
      const float distance2 = (base - event.position).squared_length();
      if( distance2 > threshold2 )
        continue;

      if( distance2 < 1.f )
        sum += _scale< TPixel >( event.value * factor );
      else
        sum += _scale< TPixel >( event.value * factor / std::sqrt( distance2 ));
    }
    return sum;
  }

  void setSource( EventSourcePtr source ) { _source = source; }
  ConstEventSourcePtr getSource() const { return _source; }

private:
  EventSourcePtr _source;

};

} // end namespace fivox

#endif
