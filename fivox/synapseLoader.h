/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_SYNAPSELOADER_H
#define FIVOX_SYNAPSELOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
/** Loads synapses to be sampled by an EventFunctor. */
class SynapseLoader : public EventSource
{
public:
    /**
    * Construct a new synapse event source.
    *
    * @param params the URIHandler object containing the parameters
    * to define the event source
    * @throw H5::exception or std::exception on error
    */
    explicit SynapseLoader( const URIHandler& params );
    virtual ~SynapseLoader();

private:
    /** @name Abstract interface implementation */
    //@{
    Vector2f _getTimeRange() const final;
    ssize_t _load( float time ) final;
    SourceType _getType() const final { return SOURCE_FRAME; }
    bool _hasEnded() const final { return true; }
    //@}

    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
