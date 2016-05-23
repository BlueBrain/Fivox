
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

#include <fivox/defines.h>
#include <fivox/event.h>            // used inline
#include <fivox/eventSource.h>      // member
#include <fivox/itk.h>
#include <lunchbox/log.h>
#include <type_traits>

namespace fivox
{

/** Samples spatial events into the given voxel. */
template< class TImage > class EventFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename TImage::SpacingType TSpacing;
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;

    EventFunctor( const Vector2f& inputRange )
        : _inputRange( inputRange )
    {}
    virtual ~EventFunctor() {}

    void setSource( EventSourcePtr source ) { _source = source; }
    ConstEventSourcePtr getSource() const { return _source; }
    EventSourcePtr getSource() { return _source; }

    /** Called before threads are starting to voxelize */
    void beforeGenerate() { if( _source ) _source->beforeGenerate(); }

    virtual TPixel operator()( const TPoint& point, const TSpacing& spacing )
        const = 0;

protected:
    TPixel _scale( const float value ) const
    {
        // scale only for output integer types
        if( std::is_floating_point< TPixel >::value )
            return value;

        const float outputMin = std::numeric_limits< TPixel >::min();
        const float outputMax = std::numeric_limits< TPixel >::max();

        const float out = ( value - _inputRange[0] ) * ( outputMax - outputMin )
                          / ( _inputRange[1] - _inputRange[0] ) + outputMin;

        // clamp to output range
        return std::max( std::min( out, outputMax ), outputMin );
    }

    const Vector2f _inputRange;
    EventSourcePtr _source;
};

}

#endif
