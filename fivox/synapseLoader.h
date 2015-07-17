/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_SYNAPSELOADER_H
#define FIVOX_SYNAPSELOADER_H

#include <fivox/eventSource.h> // base class
#include <BBP/Types.h>

namespace fivox
{
/** Loads synapses to be sampled by an EventFunctor. */
class SynapseLoader : public EventSource
{
public:
    /**
    * Construct a new synapse event source.
    *
    * @param blueconfig The Blueconfig file for the simulation
    * @param target The target to load
    * @throw H5::exception or std::exception on error
    */
    SynapseLoader( const std::string& blueconfig, const std::string& target );
    virtual ~SynapseLoader();

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
