
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 */

#ifndef FIVOX_EVENTSOURCE_H
#define FIVOX_EVENTSOURCE_H

#include <fivox/attenuationCurve.h>
#include <fivox/types.h>
#include <lunchbox/compiler.h>

namespace fivox
{
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
     *
     * @param frame The frame number to be loaded. Whether frame numbers are
     *              relative to the start time or absolute (frame 0 is at time
     *              0) depends on the actual data source.
     * @return true if the frame can be retrieved from the data source
     */
    bool load( uint32_t frame );

    /**
     * Given a timestamp, update the event source with new events to be
     * sampled.
     *
     * @param time The time stamp (ms) to be loaded.
     * @return true if the time stamp can be retrieved from the data source
     */
    bool load( float time );

    /**
     * Set the attenuation curve that will be applied to the computed events
     *
     * @param curve The attenuation curve to apply
     */
    virtual void setCurve( const AttenuationCurve& curve LB_UNUSED ) {}

    /**
     * Gets the valid frame range according to data. The valid frames are in the
     * [a, b) range
     * @return the valid frame range
     */
    Vector2ui getFrameRange() const;

    /**
     * @param frame The frame number to be checked.
     * @return Checks the frame range, if the frame is satisfying [a, b)
     * range, returns true
     */
    bool isInFrameRange( uint32_t frame );

    /**
     * @return the dt used for frame-to-time conversion in the data source,
     *         default is -1.0 and should be set accordingly in constructor.
     */
    float getDt() const;

protected:
    explicit EventSource( const URIHandler& params );

    /** @name Abstract interface */
    //@{
    /** @return the interval [a, b) in ms of available events. */
    virtual Vector2f _getTimeRange() const = 0;

    /** @sa EventSource::load( float ) */
    virtual bool _load( float time ) = 0;

    /** @return the type of this event source, needed for getFrameRange() */
    virtual SourceType _getType() const = 0;

    /**
     * @return whether all data from the source is available or it's
     * an ongoing stream, needed for getFrameRange()
     */
    virtual bool _hasEnded() const = 0;
    //@}

    /**
     * Set the dt that the datasource is using to correctly compute frame
     * number from time in load().
     *
     * This should be called by derived classes in their constructor.
     */
    void setDt( float dt );

private:
    EventSource() = delete;
    EventSource( const EventSource& ) = delete;
    EventSource& operator=( const EventSource& ) = delete;
    class Impl;
    std::unique_ptr< Impl > _impl;
};

} // end namespace fivox

#endif
