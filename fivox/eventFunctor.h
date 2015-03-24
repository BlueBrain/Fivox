/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENTFUNCTOR_H
#define FIVOX_EVENTFUNCTOR_H

#include <fivox/defines.h>
#include <fivox/eventSource.h> // member
#include <fivox/event.h>       // used inline
#include <fivox/itk.h>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace fivox
{
namespace
{
template< class T > inline T _scale( const float value )
    { return value; }
template<> inline unsigned char _scale( const float value )
    { return std::min( value * 256.f, 255.f ); }

static const bool _useCutoff = true;
static const bool _useRegion = true;
}

template< class TImage >
typename TImage::PixelType cutoffFunction(
                          const EventSource& source,
                          const typename TImage::PointType& point )
{
    static const float threshold = 50.f;
    static const float threshold2 = threshold * threshold;

    Vector3f base;
    const size_t components = std::min( point.Size(), 3u );
    for( size_t i = 0; i < components; ++i )
        base[i] = point[i];

    const AABBf region( base - Vector3f( threshold ),
                        base + Vector3f( threshold ));
    const Events& events = _useRegion ? source.findEvents( region ) :
                                        source.getEvents();
    float sum = 0.f;

    BOOST_FOREACH( const Event& event, events )
    {
        const float distance2 = (base - event.position).squared_length();
        if( _useCutoff && distance2 > threshold2 )
            continue;

        sum += event.value / distance2;
    }
    return _scale< typename TImage::PixelType >( sum );
}


/** Functor sampling spatial events into the given pixel. */
template< typename TImage > class EventFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;
    typedef typename boost::function< TPixel( const EventSource&,
                                              const TPoint& )> EventFunction;

    EventFunctor()
        : _eventFunction(&cutoffFunction<TImage>) {}
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

    EventFunction _eventFunction;
    EventSourcePtr _source;

};

} // end namespace fivox

#endif
