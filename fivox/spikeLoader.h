/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */

#ifndef FIVOX_SPIKELOADER_H
#define FIVOX_SPIKELOADER_H

#include <fivox/eventSource.h> // base class

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

private:
    /** @name Abstract interface implementation */
    //@{
    Vector2f _getTimeRange() const final;
    bool _load( float time ) final;
    SourceType _getType() const final;
    //@}

    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
