
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENTSOURCE_H
#define FIVOX_EVENTSOURCE_H

#include <fivox/attenuationCurve.h>
#include <fivox/types.h>
#include <lunchbox/compiler.h>
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

    /**
    * Given a timestamp, update the event source with new events to be
    * sampled.
    * @param time The time stamp (ms) to be loaded.
    */
    virtual void load( float time LB_UNUSED ) {}

    /**
    * Set the attenuation curve that will be applied to the computed events
    * @param curve The attenuation curve to apply
    */
    virtual void setCurve( const AttenuationCurve& curve LB_UNUSED ) {}

protected:
    EventSource();

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};

} // end namespace fivox

#endif
