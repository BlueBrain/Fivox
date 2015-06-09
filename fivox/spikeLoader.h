/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_SPIKELOADER_H
#define FIVOX_SPIKELOADER_H

#include <fivox/eventSource.h> // base class
#include <BBP/Types.h>

namespace fivox
{
namespace detail { class SpikeLoader; }

class SpikeLoader : public EventSource
{
public:
    SpikeLoader( const std::string& blueconfig, const std::string& target,
                 const std::string& spikes, const float time,
                 const float duration );
    virtual ~SpikeLoader();

    bool loadFrame( const float time, const float duration );

private:
    detail::SpikeLoader* const _impl;
};
} // end namespace fivox

#endif
