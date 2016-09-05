
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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

#ifndef FIVOX_EVENTFUNCTOR_H
#define FIVOX_EVENTFUNCTOR_H

#include <fivox/types.h>

namespace fivox
{

/** Samples spatial events into the given voxel. */
template< class TImage > class EventFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename TImage::SpacingType TSpacing;

    EventFunctor() {}
    virtual ~EventFunctor() {}

    void setEventSource( EventSourcePtr source ) { _source = source; }

    /** Called before threads are starting to voxelize */
    virtual void beforeGenerate() {}

    virtual TPixel operator()( const TPoint& point, const TSpacing& spacing )
        const = 0;

protected:
    EventSourcePtr _source;
};

}

#endif
