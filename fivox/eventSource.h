// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENTSOURCE_H
#define FIVOX_EVENTSOURCE_H

#include <fivox/event.h> // used inline
#include <vmmlib/util.hpp>
#include <boost/shared_ptr.hpp>

namespace fivox
{
/** Interface for an Event source.  */
class EventSource
{
public:
  EventSource()
      : _min( Vector3f( std::numeric_limits< float >::max( )))
      , _max( -_min )
  {}

  virtual ~EventSource() {}

  /** @return the list of events. */
  const Events& getEvents() const { return _events; }

  /** @return the minimum of the bounding box of all events. */
  const Vector3f& getMin() const { return _min; }

  /** @return the maximum of the bounding box of all events. */
  const Vector3f& getMax() const { return _max; }

  /** Clear all stored events and bounding box. */
  void clear()
  {
    _events.clear();
    _min = Vector3f( std::numeric_limits< float >::max( ));
    _max = -_min;
  }

  /** Add a new event and update the bounding box. */
  void add( const Event& event )
  {
    assert( event.position.find_min() > -std::numeric_limits< float >::max( ));
    assert( event.position.find_max() < std::numeric_limits< float >::max( ));

    _min = vmml::min( _min, event.position );
    _max = vmml::max( _max, event.position );
    _events.push_back( event );
  }

  /** Update the value of an existing event */
  void update( const size_t index, const float value )
  {
    assert( index < _events.size( ));
    _events[ index ].value = value;
  }

private:
  Events _events;
  Vector3f _min;
  Vector3f _max;
};

typedef boost::shared_ptr< EventSource > EventSourcePtr;
} // end namespace fivox

#endif
