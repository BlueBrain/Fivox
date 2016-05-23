
/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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

#ifndef FIVOX_FIELDFUNCTOR_H
#define FIVOX_FIELDFUNCTOR_H

#include <fivox/eventFunctor.h> // base class
#include <brion/types.h>

namespace fivox
{
/** Samples spatial events into the given pixel using a squared falloff. */
template< typename TImage > class FieldFunctor : public EventFunctor< TImage >
{
    typedef EventFunctor< TImage > Super;
    typedef typename Super::TPixel TPixel;
    typedef typename Super::TPoint TPoint;
    typedef typename Super::TSpacing TSpacing;

public:
    FieldFunctor( const Vector2f& inputRange )
        : Super( inputRange )
    {}
    virtual ~FieldFunctor() {}

    TPixel operator()( const TPoint& point, const TSpacing& spacing )
        const override;
};

template< class TImage > inline typename FieldFunctor< TImage >::TPixel
FieldFunctor< TImage >::operator()( const TPoint& point, const TSpacing& ) const
{
    if( !Super::_source )
        return 0;

    Vector3f base;
    const size_t components = std::min( point.Size(), 3u );
    for( size_t i = 0; i < components; ++i )
        base[i] = point[i];

    const float cutOffDistance = Super::_source->getCutOffDistance();
    const AABBf region( base - Vector3f( cutOffDistance ),
                        base + Vector3f( cutOffDistance ));
    const Events& events = Super::_source->findEvents( region );

    const float squaredCutoff = cutOffDistance * cutOffDistance;
    float sum = 0;
    for( const Event& event : events )
    {
        // OPT: do 'manual' operator- and squared_length(), vtune says it's
        // faster than using vmml vector functions
        const Vector3f distance( base.array[0] - event.position.array[0],
                                 base.array[1] - event.position.array[1],
                                 base.array[2] - event.position.array[2] );
        const float distance2( distance.array[0] * distance.array[0] +
                               distance.array[1] * distance.array[1] +
                               distance.array[2] * distance.array[2] );

        if( distance2 > squaredCutoff )
            continue;

        // If center of the voxel within the event radius, use the
        // voltage at the surface of the compartment (at 'radius' distance)
        const float contribution =
                distance2 < event.radius * event.radius ? 1.f / event.radius
                                                        : 1.f / distance2;
        sum += contribution * event.value;
    }
    return Super::_scale( sum );
}

}

#endif
