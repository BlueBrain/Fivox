
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENTSOURCE_H
#define FIVOX_EVENTSOURCE_H

#include <fivox/types.h>

#include <lunchbox/compiler.h>

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

namespace fivox
{
/**
 * Base class for an Event source.
 *
 * An event source is used by an EventFunctor to sample events for a given
 * point. Subclassing or aggregation provides the events using add() and
 * update(), and the functor accesses the data using getEvents().
 */
class EventSource : public boost::noncopyable
{
public:
    EventSource();
    virtual ~EventSource();

    /** @return the list of events. */
    const Events& getEvents() const;

    /**
    * Find all events in the given area.
    *
    * Returns a conservative set of events, may contain events outside of the
    * area, depending on the implementation.
    *
    * @param area The query bounding box.
    * @return The events contained in the area.
    */
    Events findEvents( const AABBf& area ) const;

    /** @return the bounding box of all events. */
    const AABBf& getBoundingBox() const;

    /** Clear all stored events and bounding box. */
    void clear();

    /** Add a new event and update the bounding box. */
    void add( const Event& event );

    /** Update the value of an existing event. */
    void update( const size_t index, const float value );

    /**
    * Given a frame number, update the event source with new events to be
    * sampled.
    * @param frame The frame number to be loaded.
    */
    virtual void load( uint32_t frame LB_UNUSED ) {}

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};

typedef boost::shared_ptr< EventSource > EventSourcePtr;
typedef boost::shared_ptr< const EventSource > ConstEventSourcePtr;
} // end namespace fivox

#endif
