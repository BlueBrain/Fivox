
/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     ahmet.bilgili@epfl.ch
 */

#ifndef FIVOX_ATTN_EVENTFUNCTOR_H
#define FIVOX_ATTN_EVENTFUNCTOR_H

#include <fivox/types.h>
#include <fivox/defines.h>
#include <fivox/eventSource.h> // member
#include <fivox/event.h>       // used inline
#include <fivox/itk.h>
#include <fivox/utils/attenuationCurve.h>
#include <fivox/utils/helperFunctions.h>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace fivox
{

/**
 * Compute magnitude of events using a depth attenuation curve along the Y axis.
 */
template< class TImage >
class AttenuationFunctor
{
public:

    /**
     * @param dyeAttenuationFile Attenuation curve file name.
     * @param thickness Maximum depth of the whole region.
     * @param cutOffDistance Maximum event distance.
     */
    AttenuationFunctor( const std::string& dyeAttenuationFile,
                        const float thickness,
                        const float cutOffDistance )
        : _curve( dyeAttenuationFile, thickness )
        , _cutOffDistance( cutOffDistance )
    {}

    typename TImage::PixelType attenuationFunction(
                              const EventSource& source,
                              const typename TImage::PointType& point )
    {
        const float cutOffDistance2 = _cutOffDistance * _cutOffDistance;

        Vector3f base;
        const size_t components = std::min( point.Size(), 3u );
        for( size_t i = 0; i < components; ++i )
            base[i] = point[i];

        const fivox::AABBf region( base - Vector3f( _cutOffDistance ),
                                   base + Vector3f( _cutOffDistance ));

        const fivox::Events& events = source.findEvents( region );

        float sum = 0.f;

        if( events.empty())
            return scale< typename TImage::PixelType >( sum );

        const float yMax = source.getBoundingBox().getMax()[1];
        BOOST_FOREACH( const Event& event, events )
        {
            const float distance2 = (base - event.position).squared_length();
            if( distance2 > cutOffDistance2 )
                continue;
            const float depth = yMax - event.position[1];
            sum += event.value * _curve.getAttenuation( depth ) / distance2;
        }

        return scale< typename TImage::PixelType >( sum );
    }

private:
    AttenuationCurve _curve;
    float _cutOffDistance;
};

}
#endif
