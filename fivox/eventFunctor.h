
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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

    EventFunctor(){}
    virtual ~EventFunctor() {}

    void setSource( EventSourcePtr source ) { _source = source; }
    ConstEventSourcePtr getSource() const { return _source; }
    EventSourcePtr getSource() { return _source; }

    /** Called before threads are starting to voxelize */
    virtual void beforeGenerate() {}

    virtual TPixel operator()( const TPoint& point, const TSpacing& spacing )
        const = 0;

protected:
    TPixel _scale( const float value ) const
    {
        if( std::is_floating_point< TPixel >::value )
            return value;
        static float clamped = 1.f;
        if( value > clamped )
        {
            clamped = value;
            LBINFO << "Clamping sampled value " << value << " to 1"
                   << std::endl;
        }
        return std::min( value, 1.0f ) * std::numeric_limits< TPixel >::max();
    }

    EventSourcePtr _source;
};

}

#endif
