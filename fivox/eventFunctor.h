
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_EVENTFUNCTOR_H
#define FIVOX_EVENTFUNCTOR_H

#include <fivox/defines.h>
#include <fivox/eventSource.h> // member
#include <fivox/event.h>       // used inline
#include <fivox/itk.h>
#include <fivox/eventFunctors/squaredDistanceFunctor.h> // Default functor

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

namespace fivox
{

/** Functor sampling spatial events into the given pixel. */
template< typename TImage > class EventFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;
    typedef boost::function< TPixel( const EventSource&,
                                     const TPoint& )> EventFunction;
    typedef SquaredDistanceFunctor< TImage > DefaultFunctor;

    EventFunctor()
        : _defaultFunctor( 50.0f )
        , _eventFunction( boost::bind(
                      &DefaultFunctor::fallOffFunction,
                      &_defaultFunctor, _1, _2 ))
    {}

    ~EventFunctor() {}

    bool operator!=(const EventFunctor& other) const { return !(*this == other); }
    bool operator==(const EventFunctor& other) const
        { return _eventFunction == other._eventFunction &&
                _source.get() == other._source.get(); }

    inline TPixel operator()( const TPoint& point ) const
    {
        if( !_source )
            return 0;

        return _eventFunction( *_source, point );
    }

    void setSource( EventSourcePtr source ) { _source = source; }
    ConstEventSourcePtr getSource() const { return _source; }
    EventSourcePtr getSource() { return _source; }

    void setEventFunction( const EventFunction& eventFunction )
    {
        _eventFunction = eventFunction;
    }

private:

    DefaultFunctor _defaultFunctor;
    EventFunction _eventFunction;
    EventSourcePtr _source;
};

} // end namespace fivox

#endif
