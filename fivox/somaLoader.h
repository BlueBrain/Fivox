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
    * @param params the URIHandler object containing the parameters
    * to define the event source
    * @throw H5::exception or std::exception on error
    */
    explicit SomaLoader( const URIHandler& params );
    virtual ~SomaLoader();

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
