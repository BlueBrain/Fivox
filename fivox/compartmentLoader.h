/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 */

#ifndef FIVOX_COMPARTMENTLOADER_H
#define FIVOX_COMPARTMENTLOADER_H

#include <fivox/eventSource.h> // base class
#include <fivox/uriHandler.h>

namespace fivox
{
/** Loads compartment report data to be sampled by an EventFunctor. */
class CompartmentLoader : public EventSource
{
public:
    /**
    * Construct a new compartment event source.
    *
    * @param params the URIHandler object containing the parameters
    * to define the event source
    * @throw H5::exception or std::exception on error
    */
    explicit CompartmentLoader( const URIHandler& params );
    virtual ~CompartmentLoader(); //!< Destruct this compartment event source

    /**
    * Load the data at the given timestamp
    * @param time the timestamp of interest
    * @return true if the time stamp can be retrieved from the data source
    */
    bool load( float time ) final;

    /**
    * Load a new frame, based on the duration defined for each timestep (dt)
    * @param frame The frame number relative to the starting time of the report.
    * @return true if the frame can be retrieved from the data source
    */
    bool load( uint32_t frame ) final;

    /**
     * Gets the valid frame range according to data. If the simulation
     * reporting starts at t_0, frame 0 is available for any time stamp
     * t > t_0. If the reporting delta is d, at any t in (t_0, t_0 + d],
     * this function will return (0, 1)
     * @return the valid frame range [a,b)
     */
    Vector2ui getFrameRange() final;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
