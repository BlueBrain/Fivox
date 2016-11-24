
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

#include <fivox/api.h>
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
    FIVOX_API virtual ~EventSource();

    /**
     * Resize the underlying event structure to the specified size, and
     * initialize all event attributes to 0
     *
     * @param numEvents the number of events that the EventSource will hold
     */
    FIVOX_API void resize( size_t numEvents );

    /**
     * Get a reference to the value of an event contained in the EventSource
     * by its index.
     *
     * @param index Index of the event whose value will be returned
     * @return value of the event stored in the EventSource with the
     * specified index.
     */
    FIVOX_API float& operator[]( size_t index );

    /** @return the number of events */
    FIVOX_API size_t getNumEvents() const;

    /** @return a const pointer to the X coordinates of the event positions */
    FIVOX_API const float* getPositionsX() const;

    /** @return a const pointer to the Y coordinates of the event positions */
    FIVOX_API const float* getPositionsY() const;

    /** @return a const pointer to the Z coordinates of the event positions */
    FIVOX_API const float* getPositionsZ() const;

    /** @return a const pointer to the events' inverse radii (1/radius).
     *  This helps speeding up common computations later on, e.g. functors. */
    FIVOX_API const float* getRadii() const;

    /** @return a const pointer to the events' values */
    FIVOX_API const float* getValues() const;

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
    FIVOX_API EventValues findEvents( const AABBf& area ) const;

    /**
     * Set bounding box of upcoming events. This overwrites any existing
     * bounding box. It can be used to set a bounding box before
     * loading/updating any events.
     */
    FIVOX_API void setBoundingBox( const AABBf& boundingBox );

    /** @return the bounding box of all events. */
    FIVOX_API const AABBf& getBoundingBox() const;

    /**
     * @return the cutoff distance currently set (50.0 by default)
     */
    FIVOX_API float getCutOffDistance() const;

    /**
     * Update attributes of the event specified by the index. Update also the
     * bounding box to include the new position. The specified index should
     * be smaller than the size used in resize()
     * Not thread safe.
     *
     * @param i the index of the event that will be updated
     * @param pos the event position
     * @param rad the event radius
     * @param val the event value, set to 0 if not specified
     */
    FIVOX_API void update( size_t i, const Vector3f& pos, float rad, float val = 0.f );

    /**
     * @internal Called before data is read. Not thread safe.
     * Build an RTree so it can be used from findEvents() (depends
     * on boost::geometry)
     */
    FIVOX_API void buildRTree();

    /**
     * Given a frame number, update the event source with new events to be
     * sampled.
     *
     * @param frame The frame number to be loaded. Whether frame numbers are
     *              relative to the start time or absolute (frame 0 is at time
     *              0) depends on the actual data source.
     * @return true if the frame is within the range of the data source.
     */
    FIVOX_API bool setFrame( uint32_t frame );

    /**
     * Given a timestamp, update the event source with new events to be sampled.
     *
     * @param time The time stamp (ms) to be loaded.
     */
    FIVOX_API void setTime( float time );

    /**
     * Gets the valid frame range according to data. The valid frames are in the
     * [a, b) range
     * @return the valid frame range
     */
    FIVOX_API Vector2ui getFrameRange() const;

    /**
     * @param frame The frame number to be checked.
     * @return Checks the frame range, if the frame is satisfying [a, b)
     * range, returns true
     */
    FIVOX_API bool isInFrameRange( uint32_t frame );

    /**
     * @return the dt used for frame-to-time conversion in the data source,
     *         default is -1.0 and should be set accordingly in constructor.
     */
    FIVOX_API float getDt() const;

    /** @return the current time from setTime() in milliseconds. */
    FIVOX_API float getCurrentTime() const;

    /**
     * Load and update events for the given chunks of the data source.
     *
     * @param chunkIndex the chunk to start loading from
     * @param numChunks the number of chunks to load into memory
     * @return the number of updated events, or -1 if the load failed.
     * @throw std::runtime_error if numChunks is 0.
     * @throw std::out_of_range if chunkIndex and/or numChunks are out of range.
     */
    FIVOX_API ssize_t load( size_t chunkIndex, size_t numChunks );

    /**
     * Load and update all events of the current frame.
     *
     * @return the number of updated events, or -1 if the load failed.
     */
    FIVOX_API ssize_t load();

    /** @return the maximum number of chunks provided by the data source. */
    FIVOX_API size_t getNumChunks() const;

    /**
     * Read events from the specified file. It will first try to read it as a
     * binary file, checking the expected magic number; if not found, it will
     * read it as an ASCII, in the expected format (see specification).
     *
     * The contents of the file will be used to set the events in the
     * EventSource.
     *
     * @param filename path of the file to read the events from.
     * @return true if the events were read correctly, false otherwise.
     * @throw std::runtime_error if the specified file is empty.
     * @throw std::runtime_error if the binary file version is incompatible.
     * @throw std::runtime_error if the number of events to load from an ASCII
     * file is 0, or not specified.
     * @throw std::runtime_error if any of the events being read is ill-formed.
     */
    FIVOX_API bool read( const std::string& filename );

    /**
     * Write events to the specified file. It is possible to specify the format
     * of the output (binary or ASCII, see specification for reference).
     *
     * @param filename path of the file to write the events to.
     * @param format file format in which the event file will be written
     * (EventFileFormat::ascii and EventFileFormat::binary are supported).
     * @return true if the file was succesfully written, false otherwise.
     */
    FIVOX_API bool write( const std::string& filename,
                          EventFileFormat format ) const;

protected:
    explicit EventSource( const URIHandler& params );

    EventSource() = delete;
    EventSource( const EventSource& ) = delete;
    EventSource& operator=( const EventSource& ) = delete;

    /** @name Abstract interface */
    //@{
    /** @return the interval [a, b) in ms of available events. */
    virtual Vector2f _getTimeRange() const = 0;

    /** @sa EventSource::load( size_t, size_t ) */
    virtual ssize_t _load( size_t chunkIndex, size_t numChunks ) = 0;

    /** @return the type of this event source, needed for getFrameRange() */
    virtual SourceType _getType() const = 0;

    /**
     * @return whether all data from the source is available or it's
     * an ongoing stream, needed for getFrameRange()
     */
    virtual bool _hasEnded() const = 0;

    /** @sa EventSource::getNumChunks() */
    virtual size_t _getNumChunks() const = 0;
    //@}

    /**
     * Set the dt that the datasource is using to correctly compute frame
     * number from time in load().
     *
     * This should be called by derived classes in their constructor.
     */
    void setDt( float dt );

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};

} // end namespace fivox

#endif
