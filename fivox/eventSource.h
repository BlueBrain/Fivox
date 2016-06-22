
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
 * An event source is used by an EventFunctor to sample events for a given point
 * at a given time. Subclassing provides the events using resize() and update(),
 * and the functor accesses the data using getter methods for each of the event
 * fields.
 */
class EventSource
{
public:
    virtual ~EventSource();

    /**
     * Resize the underlying event structure to the specified size, and
     * initialize all event attributes to 0
     *
     * @param numEvents the number of events that the EventSource will hold
     */
    void resize( size_t numEvents );

    /**
     * Get a reference to the value of an event contained in the EventSource
     * by its index.
     *
     * @param index Index of the event whose value will be returned
     * @return value of the event stored in the EventSource with the
     * specified index.
     */
    float& operator[]( size_t index );

    /** @return the number of events */
    size_t getNumEvents() const;

    /** @return a const pointer to the X coordinates of the event positions */
    const float* getPositionsX() const;

    /** @return a const pointer to the Y coordinates of the event positions */
    const float* getPositionsY() const;

    /** @return a const pointer to the Z coordinates of the event positions */
    const float* getPositionsZ() const;

    /** @return a const pointer to the events' inverse radii (1/radius).
     *  This helps speeding up common computations later on, e.g. functors. */
    const float* getRadii() const;

    /** @return a const pointer to the events' values */
    const float* getValues() const;

    /**
     * Find all events in the given area.
     *
     * Returns a vector of values corresponding to a conservative set of events,
     * may contain events outside of the area, depending on the implementation.
     *
     * @param area The query bounding box.
     * @return The values of the events contained in the area. Empty if no RTree
     * available (depends on boost::geometry)
     */
    EventValues findEvents( const AABBf& area ) const;

    /** @return the bounding box of all events. */
    const AABBf& getBoundingBox() const;

    /**
     * @return the cutoff distance currently set (50.0 by default)
     */
    float getCutOffDistance() const;

    /**
     * Update attributes of the event specified by the index. Update also the
     * bounding box to include the new position. The specified index should
     * be smaller than the size used in resize()
     * Not thread safe.
     */
    void update( const size_t i, const Vector3f pos,
                 const float rad, const float val = 0.f );

    /**
     * @internal Called before data is read. Not thread safe.
     * Build an RTree so it can be used from findEvents() (depends
     * on boost::geometry)
     */
    void buildRTree();

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

    /**
     * @sa EventSource::load( float )
     * @return the number of updated events, or -1 if the load failed.
     */
    virtual ssize_t _load( float time ) = 0;

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
