
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
    FieldFunctor()
        : Super()
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

    const float cutOffDistance = Super::_source->getCutOffDistance();

    const size_t size = Super::_source->getNumEvents();
    // By using the restrict keyword, we specify that the object will only
    // be accessed by the declared pointer, which helps for the optimization
    const float* __restrict__ posx = Super::_source->getPositionsX();
    const float* __restrict__ posy = Super::_source->getPositionsY();
    const float* __restrict__ posz = Super::_source->getPositionsZ();
    const float* __restrict__ radii = Super::_source->getRadii();
    const float* __restrict__ values = Super::_source->getValues();

    const float px( point[0] ), py( point[1] ), pz( point[2] );

    // Compute directly the inverted value to gain performance in the for loop
    const float squaredCutoff = 1.f / ( cutOffDistance * cutOffDistance );
    float voltage1( 0.f ), voltage2( 0.f );

    // Tell the compiler that memory accesses are aligned (done in
    // EventSource::resize) so it is able to make optimizations
    #pragma vector aligned
    for( size_t i = 0; i < size; ++i )
    {
        const float distanceX = px - posx[i];
        const float distanceY = py - posy[i];
        const float distanceZ = pz - posz[i];

        const float distance2( 1.f / ( distanceX * distanceX +
                                       distanceY * distanceY +
                                       distanceZ * distanceZ ));

        // Comparison is inverted, as we are using the reciprocal values
        if( distance2 < squaredCutoff )
            continue;

        const float value( values[i] );

        // If center of the voxel within the event radius, use the
        // voltage at the surface of the compartment (at 'radius' distance)
        const float radius( radii[i] );
        // Comparison is inverted, as we are using the reciprocal values
        // (radius is already inverted from the loader)
        if( distance2 > radius * radius )
            voltage1 += value * radius; // mV
        else
            voltage2 += value * distance2; // mV
    }
    return voltage1 + voltage2;
}

}

#endif
