
/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Ahmet.Bilgili@epfl.ch
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_ATTENUATIONFUNCTOR_H
#define FIVOX_ATTENUATIONFUNCTOR_H

#include <fivox/eventFunctor.h>      // base class
#include <fivox/attenuationCurve.h> // member

namespace fivox
{
/** Calculate the value of a point using an attenuation curve. */
template< class TImage >
class AttenuationFunctor : public fivox::EventFunctor< TImage >
{
    typedef fivox::EventFunctor< TImage > Super;

public:
    AttenuationFunctor() {}
    virtual ~AttenuationFunctor() {}

    void setCurve( const AttenuationCurve& curve ) { _curve = curve; }

    typename Super::TPixel operator()( const typename Super::TPoint& point )
        const override
    {
        if( !Super::_source )
            return 0;

        const float cutOffDistance2 = Super::_cutOffDistance *
                                      Super::_cutOffDistance;
        Vector3f base;
        const size_t components = std::min( point.Size(), 3u );
        for( size_t i = 0; i < components; ++i )
            base[i] = point[i];

        const AABBf region( base - Vector3f( Super::_cutOffDistance ),
                            base + Vector3f( Super::_cutOffDistance ));
        const Events& events = Super::_source->findEvents( region );

        float sum = 0.f;
        const float yMax = Super::_source->getBoundingBox().getMax()[1];

        BOOST_FOREACH( const Event& event, events )
        {
            const float distance2 = (base - event.position).squared_length();
            if( distance2 > cutOffDistance2 )
                continue;

            const float depth = yMax - event.position[1];
            sum += event.value * _curve.getAttenuation( depth ) / distance2;
        }

        return Super::_scale( sum );
    }

private:
    AttenuationCurve _curve;
};

}

#endif
