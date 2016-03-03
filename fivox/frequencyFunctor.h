
/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
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

#ifndef FIVOX_FREQUENCYFUNCTOR_H
#define FIVOX_FREQUENCYFUNCTOR_H

#include <fivox/eventFunctor.h> // base class

namespace fivox
{

/** Projects maximum frequency of events into the given voxel. */
template< typename TImage >
class FrequencyFunctor : public EventFunctor< TImage >
{
    typedef EventFunctor< TImage > Super;
    typedef typename Super::TPixel TPixel;
    typedef typename Super::TPoint TPoint;
    typedef typename Super::TSpacing TSpacing;

public:
    FrequencyFunctor( const fivox::Vector2f& inputRange )
        : Super( inputRange )
    {}
    virtual ~FrequencyFunctor() {}

    TPixel operator()( const TPoint& point, const TSpacing& spacing )
        const override;
};

template< class TImage > inline typename FrequencyFunctor< TImage >::TPixel
FrequencyFunctor< TImage >::operator()( const TPoint& itkPoint,
                                        const TSpacing& itkSpacing ) const
{
    if( !Super::_source )
        return 0;

    Vector3f point;
    Vector3f spacing_2;
    const size_t components = std::min( itkPoint.Size(), 3u );
    for( size_t i = 0; i < components; ++i )
    {
        point[i] = itkPoint[i];
        spacing_2[i] = itkSpacing[i] * 0.5;
    }

    const AABBf region( point - spacing_2, point + spacing_2 );
    const Events& events = Super::_source->findEvents( region );

    float sum = 0.f;
    for( const Event& event : events )
        sum = std::max( sum, event.value );

    return Super::_scale( sum );
}

}

#endif
