/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */
#ifndef FIVOX_VSDLOADER_H
#define FIVOX_VSDLOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
/**
 * Loads report data for voltage sensitive dye sampling.
 *
 * Expects an 'area' full compartment report with a single time step containing
 * the surface areas of each compartment, and a 'voltages' soma report
 * containing the soma voltages for the given time step.
 */
class VSDLoader : public EventSource
{
public:
    /**
    * Construct a new vsd event source.
    *
    * @param blueconfig The Blueconfig file for the simulation
    * @param target The target to load
    * @param dt The duration of the time step
    * @throw H5::exception or std::exception on error
    */
    VSDLoader( const std::string& blueconfig, const std::string& target,
               float dt );

    virtual ~VSDLoader(); //!< Destruct this vsd event source

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
