/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 */
#ifndef FIVOX_SOMALOADER_H
#define FIVOX_SOMALOADER_H

#include <fivox/eventSource.h> // base class
#include <fivox/uriHandler.h>

namespace fivox
{
/** Loads soma report data to be sampled by an EventFunctor. */
class SomaLoader : public EventSource
{
public:
    /**
    * Construct a new soma event source.
    *
    * @param params the URIHandler object containing the parameters
    * to define the event source
    * @throw H5::exception or std::exception on error
    */
    explicit SomaLoader( const URIHandler& params );
    virtual ~SomaLoader();

    /**
    * Load the data at the given timestamp
    * @param time the timestamp of interest
    * @return true if the time stamp can be retrieved from the data source
    */
    bool load( float time ) final;

    /**
    * Load a new frame, based on the duration defined for each timestep (dt)
    * @param frame The frame number to be loaded
    * @return true if the frame can be retrieved from the data source
    */
    bool load( uint32_t frame ) final;

    /**
     * Gets the valid frame range according to data. The valid frames are in the
     * [a, b) range
     * @return the valid frame range
     */
    Vector2ui getFrameRange() final;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
