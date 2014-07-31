// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "eventSource.h"
#include "event.h"

#include <lunchbox/log.h>

#if USE_BOOST_GEOMETRY
#  include <boost/foreach.hpp>
#  include <boost/geometry.hpp>
#  include <boost/geometry/geometries/box.hpp>
#  include <boost/geometry/geometries/point.hpp>
#  include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point< float, 3, bg::cs::cartesian > Point;
typedef bg::model::box< Point > Box;
typedef std::pair< Point, size_t > Value;

static const size_t maxElemInNode = 64;
static const size_t minElemInNode = 16;
#endif

namespace fivox
{
namespace detail
{
class EventSource
{
public:

  Events events;
  AABBf boundingBox;
#if USE_BOOST_GEOMETRY
  bgi::rtree< Value, bgi::rstar< maxElemInNode, minElemInNode > > rtree;
#endif
};
}

EventSource::EventSource()
    : _impl( new detail::EventSource )
{
}

EventSource::~EventSource()
{
  delete _impl;
}

const Events& EventSource::getEvents() const
{
  return _impl->events;
}

Events EventSource::findEvents( const AABBf& area LB_UNUSED ) const
{
#ifdef USE_BOOST_GEOMETRY
  const Vector3f& p1 = area.getMin();
  const Vector3f& p2 = area.getMax();
  const Box query( Point( p1[0], p1[1], p1[2] ), Point( p2[0], p2[1], p2[2] ));

  std::vector< Value > hits;
  _impl->rtree.query( bgi::intersects( query ), std::back_inserter( hits ));

  Events events;
  BOOST_FOREACH( const Value& value, hits )
    events.push_back( _impl->events[ value.second ] );
  return events;
#else
  static bool first = true;
  if( first )
  {
    LBWARN << "slow path: boost/geometry not available for findEvents"
           << std::endl;
    first = false;
  }
  return _impl->events;
#endif
}

const AABBf& EventSource::getBoundingBox() const
{
  return _impl->boundingBox;
}

void EventSource::clear()
{
  _impl->events.clear();
  _impl->boundingBox.setEmpty();
}

void EventSource::add( const Event& event )
{
#ifdef USE_BOOST_GEOMETRY
  const Point point( event.position[0], event.position[1], event.position[2] );
  _impl->rtree.insert( std::make_pair( point, _impl->events.size( )));
#endif

  _impl->boundingBox.merge( event.position );
  _impl->events.push_back( event );
}

void EventSource::update( const size_t index, const float value )
{
  assert( index < _impl->events.size( ));
  _impl->events[ index ].value = value;
}

}
