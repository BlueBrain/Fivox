/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 */
#ifndef FIVOX_SOMALOADER_H
#define FIVOX_SOMALOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
/** Loads soma report data to be sampled by an EventFunctor. */
class SomaLoader : public EventSource
{
public:
    /**
    * Construct a new soma event source.
    *
    * @param blueconfig The Blueconfig file for the simulation
    * @param target The target to load
    * @param report The name of the report to use, 'soma' if empty
    * @param dt The duration of the timestep. If -1, it takes the
    * one from the report as the default
    * @throw H5::exception or std::exception on error
    */
    SomaLoader( const std::string& blueconfig, const std::string& target,
                const std::string& report, float dt );
    virtual ~SomaLoader();

    /**
    * Load the data at the given timestamp
    * @param time the timestamp of interest
    */
    void load( float time );

    /**
    * Load a new frame, based on the duration defined for each timestep (dt)
    * @param frame The frame number to be loaded
    */
    void load( uint32_t frame );

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
