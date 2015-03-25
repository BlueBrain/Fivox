
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_SQUAREDDISTANCE_EVENTFUNCTOR_H
#define FIVOX_SQUAREDDISTANCE_EVENTFUNCTOR_H

#include <fivox/types.h>
#include <fivox/defines.h>
#include <fivox/eventSource.h> // member
#include <fivox/event.h>       // used inline
#include <fivox/itk.h>
#include <fivox/utils/helperFunctions.h>

#include <boost/foreach.hpp>

namespace fivox
{

/**
 * Compute the effect of events considering the given cut off distance.
 */
template< class TImage >
class SquaredDistanceFunctor
{
public:

    /**
     * @param cutOffDistance Maximum event distance.
     */
    SquaredDistanceFunctor( const float cutOffDistance )
        :  _cutOffDistance( cutOffDistance )
    {}

    typename TImage::PixelType fallOffFunction (
                              const EventSource& source,
                              const typename TImage::PointType& point )
    {

        const float cutOffDistance2 = _cutOffDistance * _cutOffDistance;

        Vector3f base;
        const size_t components = std::min( point.Size(), 3u );
        for( size_t i = 0; i < components; ++i )
            base[i] = point[i];

        const AABBf region( base - Vector3f( _cutOffDistance ),
                            base + Vector3f( _cutOffDistance ));
        const Events& events = source.findEvents( region );

        float sum = 0.f;

        BOOST_FOREACH( const Event& event, events )
        {
            const float distance2 = (base - event.position).squared_length();
            if( distance2 > cutOffDistance2 )
                continue;

            sum += event.value / cutOffDistance2;
        }
        return scale< typename TImage::PixelType >( sum );
    }
private:

    float _cutOffDistance;
};

}
#endif // FIVOX_SQUAREDDISTANCE_EVENTFUNCTOR_H
