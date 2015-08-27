/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */

#ifndef FIVOX_SPIKELOADER_H
#define FIVOX_SPIKELOADER_H

#include <fivox/eventSource.h> // base class
#include <fivox/uriHandler.h>

namespace fivox
{
/** Loads spike report data to be sampled by an EventFunctor. */
class SpikeLoader : public EventSource
{
public:
    /**
    * Construct a new spike event source.
    *
    * @param params the URIHandler object containing the parameters
    * to define the event source
    * @throw H5::exception or std::exception on error
    */
    explicit SpikeLoader( const URIHandler& params );
    virtual ~SpikeLoader();

    /**
    * Load the data at the given timestamp
    * @param time the timestamp of interest
    * @return true if the time stamp can be retrieved from the data source
    */
    bool load( float time ) final;

    /**
    * Load a new frame, based on the duration defined for each timestep (dt)
    * @param frame The absolute frame number to be loaded
    * @return true if the frame can be retrieved from the data source
    */
    bool load( uint32_t frame ) final;

    /**
     * Gets the valid frame range according to data. For spike data, each
     * frame represents a time interval [t, t + dt). Frames become available
     * every dt, the first one being reported at t = dt, that is, in the
     * range [0,dt) no full frame has been completed yet and the
     * function would return (0, 0).
     * @return the valid frame range [a,b)
     */
    Vector2ui getFrameRange() final;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
