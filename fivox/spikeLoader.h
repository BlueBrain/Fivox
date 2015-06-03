/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */

#ifndef FIVOX_SPIKELOADER_H
#define FIVOX_SPIKELOADER_H

#include <fivox/eventSource.h> // base class
#include <BBP/Types.h>

namespace fivox
{
/** Loads spike report data to be sampled by an EventFunctor. */
class SpikeLoader : public EventSource
{
public:
    /**
    * Construct a new spike event source.
    *
    * @param blueconfig The Blueconfig file for the simulation
    * @param target The target to load
    * @param spikes URI to access spikes data from. If not specified, it takes
    * the one in the BlueConfig file (SpikesPath)
    * @param dt The duration of the time step
    * @param duration The duration of the time window to be sampled at each
    * frame
    * @throw H5::exception or std::exception on error
    */
    SpikeLoader( const std::string& blueconfig, const std::string& target,
                 const std::string& spikes, float dt, float duration );
    virtual ~SpikeLoader();

    /**
    * Load a new frame, based on the duration defined for each time step (dt)
    * @param frame The frame number to be loaded
    */
    void load( uint32_t frame );

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
