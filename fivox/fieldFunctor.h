
/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_FIELDFUNCTOR_H
#define FIVOX_FIELDFUNCTOR_H

#include <fivox/eventFunctor.h> // base class

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
    FieldFunctor() : _cutOffDistance( 50. ) {}
    virtual ~FieldFunctor() {}

    void setCutOffDistance( const float distance )
        { _cutOffDistance = distance; }

    float getKernelSize() override { return _cutOffDistance; }

    TPixel operator()( const TPoint& point, const TSpacing& spacing )
        const override;

private:
    float _cutOffDistance;
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

    const AABBf region( base - Vector3f( _cutOffDistance ),
                        base + Vector3f( _cutOffDistance ));
    const Events& events = Super::_source->findEvents( region );

    float sum = 0.f;
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
        if( distance2 > 1. )
            sum += event.value / distance2;
        else
            sum += event.value;
    }

    return Super::_scale( sum );
}

}

#endif
