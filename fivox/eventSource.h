// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENTSOURCE_H
#define FIVOX_EVENTSOURCE_H

#include <fivox/types.h>
#include <boost/shared_ptr.hpp>

namespace fivox
{
namespace detail { class EventSource; }

/**
 * Base class for an Event source.
 *
 * An event source is used by an EventFunctor to sample events for a given
 * point. Subclassing or aggregation provides the events using add() and
 * update(), and the functor accesses the data using getEvents().
 */
class EventSource
{
public:
  EventSource();
  virtual ~EventSource();

  /** @return the list of events. */
  const Events& getEvents() const;

  /** @return the bounding box of all events. */
  const AABBf& getBoundingBox() const;

  /** Clear all stored events and bounding box. */
  void clear();

  /** Add a new event and update the bounding box. */
  void add( const Event& event );

  /** Update the value of an existing event */
  void update( const size_t index, const float value );

private:
  detail::EventSource* const _impl;
};

typedef boost::shared_ptr< EventSource > EventSourcePtr;
typedef boost::shared_ptr< const EventSource > ConstEventSourcePtr;
} // end namespace fivox

#endif
