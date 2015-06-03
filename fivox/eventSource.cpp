
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */

#include "eventSource.h"
#include "event.h"

#include <lunchbox/log.h>

#if USE_BOOST_GEOMETRY
#  include <lunchbox/lock.h>
#  include <lunchbox/scopedMutex.h>
#  include <boost/foreach.hpp>
#  include <boost/geometry.hpp>
#  include <boost/geometry/geometries/box.hpp>
#  include <boost/geometry/geometries/point.hpp>
#  include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point< float, 3, bg::cs::cartesian > Point;
typedef bg::model::box< Point > Box;
typedef std::pair< Point, size_t > Value;
typedef std::vector< Value > Values;

static const size_t maxElemInNode = 64;
static const size_t minElemInNode = 16;
#endif

namespace fivox
{
class EventSource::Impl
{
public:

    Events events;
    AABBf boundingBox;
#if USE_BOOST_GEOMETRY
    typedef bgi::rtree< Value, bgi::rstar< maxElemInNode, minElemInNode > > RTree;
    RTree rtree;

    void rebuildRTree()
    {
        lunchbox::ScopedWrite mutex( _rtreeLock );
        if( !rtree.empty( ))
            return;

        Values values;
        values.reserve( events.size( ));

        size_t i = 0;
        BOOST_FOREACH( const Event& event, events )
        {
            const Point point( event.position[0], event.position[1],
                    event.position[2] );
            values.push_back( std::make_pair( point, i++ ));
        }

        RTree rt( values.begin(), values.end( ));
        rtree = boost::move( rt );
    }

private:
    lunchbox::Lock _rtreeLock;
#endif
};

EventSource::EventSource()
    : _impl( new EventSource::Impl )
{}

EventSource::~EventSource()
{}

const Events& EventSource::getEvents() const
{
    return _impl->events;
}

Events EventSource::findEvents( const AABBf& area LB_UNUSED ) const
{
#ifdef USE_BOOST_GEOMETRY
    _impl->rebuildRTree();

    const Vector3f& p1 = area.getMin();
    const Vector3f& p2 = area.getMax();
    const Box query( Point( p1[0], p1[1], p1[2] ), Point( p2[0], p2[1], p2[2] ));

    std::vector< Value > hits;
    _impl->rtree.query( bgi::intersects( query ), std::back_inserter( hits ));

    Events events;
    BOOST_FOREACH( const Value& value, hits )
            events.push_back( _impl->events[ value.second ] );
    return events;
#else
    static bool first = true;
    if( first )
    {
        LBWARN << "slow path: boost/geometry not available for findEvents"
               << std::endl;
        first = false;
    }
    return _impl->events;
#endif
}

const AABBf& EventSource::getBoundingBox() const
{
    return _impl->boundingBox;
}

void EventSource::clear()
{
    _impl->events.clear();
    _impl->boundingBox.setEmpty();
}

void EventSource::add( const Event& event )
{
#ifdef USE_BOOST_GEOMETRY
    _impl->rtree.clear();
#endif

    _impl->boundingBox.merge( event.position );
    _impl->events.push_back( event );
}

void EventSource::update( const size_t index, const float value )
{
    assert( index < _impl->events.size( ));
    _impl->events[ index ].value = value;
}

}
