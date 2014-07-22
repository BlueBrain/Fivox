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
  {}

  virtual ~EventSource() {}

  /** @return the list of events. */
  const Events& getEvents() const { return _events; }

  /** @return the bounding box of all events. */
  const AABBf& getBoundingBox() const { return _boundingBox; }

  /** Clear all stored events and bounding box. */
  void clear()
  {
    _events.clear();
    _boundingBox.setEmpty();
  }

  /** Add a new event and update the bounding box. */
  void add( const Event& event )
  {
    _boundingBox.merge( event.position );
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
  AABBf _boundingBox;
};

typedef boost::shared_ptr< EventSource > EventSourcePtr;
typedef boost::shared_ptr< const EventSource > ConstEventSourcePtr;
} // end namespace fivox

#endif
