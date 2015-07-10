
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_EVENTFUNCTOR_H
#define FIVOX_EVENTFUNCTOR_H

#include <fivox/defines.h>
#include <fivox/event.h>       // used inline
#include <fivox/eventSource.h> // member
#include <fivox/itk.h>

#include <boost/type_traits/is_floating_point.hpp>

namespace fivox
{

/** Samples spatial events into the given pixel using a squared falloff. */
template< typename TImage > class EventFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;

    EventFunctor() : _cutOffDistance( 50.0f ) {}
    virtual ~EventFunctor() {}

    virtual TPixel operator()( const TPoint& point ) const;

    void setSource( EventSourcePtr source ) { _source = source; }
    ConstEventSourcePtr getSource() const { return _source; }
    EventSourcePtr getSource() { return _source; }

    void setCutOffDistance( const float distance )
        { _cutOffDistance = distance; }

protected:
    float _cutOffDistance;
    EventSourcePtr _source;

    TPixel _scale( const float value ) const
    {
        if( boost::is_floating_point< TPixel >::value )
            return value;

        return std::min( value, 1.0f ) * std::numeric_limits< TPixel >::max();
    }
};

template< class TImage > inline typename EventFunctor< TImage >::TPixel
EventFunctor< TImage >::operator()( const TPoint& point ) const
{
    if( !_source )
        return 0;

    const float cutOffDistance2 = _cutOffDistance * _cutOffDistance;

    Vector3f base;
    const size_t components = std::min( point.Size(), 3u );
    for( size_t i = 0; i < components; ++i )
        base[i] = point[i];

    const AABBf region( base - Vector3f( _cutOffDistance ),
                        base + Vector3f( _cutOffDistance ));
    const Events& events = _source->findEvents( region );

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
        if( distance2 < cutOffDistance2 )
            sum += event.value / distance2;
    }

    return _scale( sum );
}

}

#endif
