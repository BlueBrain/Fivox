/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef FIVOX_TESTLOADER_H
#define FIVOX_TESTLOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
/**
 * Create a set of dummy events, arranged in a vertical straight line,
 * to be sampled by an EventFunctor.
 */
class TestLoader : public EventSource
{
public:
    /**
    * Construct a new test event source.
    *
    * @param params the URIHandler object containing the parameters
    * to define the event source
    */
    explicit TestLoader( const URIHandler& params );
    virtual ~TestLoader();

private:
    /** @name Abstract interface implementation */
    //@{
    Vector2f _getTimeRange() const final;
    ssize_t _load( float time ) final;
    SourceType _getType() const final { return SourceType::frame; }
    bool _hasEnded() const final { return true; }
    //@}

    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
