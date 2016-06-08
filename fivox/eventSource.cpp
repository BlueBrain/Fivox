
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
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

#include "eventSource.h"
#include "event.h"
#include "uriHandler.h"

#include <lunchbox/atomic.h>
#include <lunchbox/log.h>

#ifdef USE_BOOST_GEOMETRY
#  include <lunchbox/lock.h>
#  include <lunchbox/scopedMutex.h>
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
    Impl( const URIHandler& params )
        : dt( params.getDt( ))
        , currentTime( -1.f )
        , cutOffDistance( params.getCutoffDistance( ))
    {}

    float dt;
    float currentTime;
    float cutOffDistance;
    Events events;
    AABBf boundingBox;
#ifdef USE_BOOST_GEOMETRY
    typedef bgi::rtree< Value, bgi::rstar< maxElemInNode, minElemInNode > > RTree;
    RTree rtree;

    void rebuildRTree()
    {
        if( !rtree.empty( ))
            return;

        LBINFO << "Building rtree for " << events.size() << " events"
               << std::endl;
        Values values;
        values.reserve( events.size( ));

        size_t i = 0;
        for( const Event& event : events )
        {
            const Point point( event.position[0], event.position[1],
                               event.position[2] );
            values.push_back( std::make_pair( point, i++ ));
        }

        RTree rt( values.begin(), values.end( ));
        rtree = boost::move( rt );
        LBINFO << " done" << std::endl;
    }
#endif
};

EventSource::EventSource( const URIHandler& params )
    : _impl( new EventSource::Impl( params ))
{}

EventSource::~EventSource()
{}

const Events& EventSource::getEvents() const
{
    return _impl->events;
}

Event& EventSource::operator[]( const size_t index )
{
    assert( index < _impl->events.size( ));
    return _impl->events[index];
}

Events EventSource::findEvents( const AABBf& area LB_UNUSED ) const
{
#ifdef USE_BOOST_GEOMETRY
    if( !_impl->rtree.empty( ))
    {
        const Vector3f& p1 = area.getMin();
        const Vector3f& p2 = area.getMax();
        const Box query( Point( p1[0], p1[1], p1[2] ),
                         Point( p2[0], p2[1], p2[2] ));

        static lunchbox::a_ssize_t maxHits( 0 );
        std::vector< Value > hits;
        hits.reserve( maxHits );
        _impl->rtree.query( bgi::intersects( query ), std::back_inserter( hits ));
        maxHits = std::max( size_t(maxHits), hits.size( ));

        Events events;
        events.reserve( hits.size( ));
        for( const Value& value : hits )
        {
            const Event& val = _impl->events[ value.second ];
            if( val.value != VALUE_UNSET )
                events.push_back( val );
        }
        return events;
    }
#endif

    static bool first = true;
    if( first )
    {
        LBWARN << "slow path: rtree acceleration not available for findEvents"
               << std::endl;
        first = false;
    }
    return _impl->events;
}

const AABBf& EventSource::getBoundingBox() const
{
    return _impl->boundingBox;
}

float EventSource::getCutOffDistance() const
{
    return _impl->cutOffDistance;
}

void EventSource::clear()
{
    _impl->events.clear();
    _impl->boundingBox.reset();
}

void EventSource::add( const Event& event )
{
#ifdef USE_BOOST_GEOMETRY
    _impl->rtree.clear();
#endif

    _impl->boundingBox.merge( event.position );
    _impl->events.push_back( event );
}

void EventSource::beforeGenerate()
{
#ifdef USE_BOOST_GEOMETRY
    _impl->rebuildRTree();
#endif
}

bool EventSource::load( const uint32_t frame )
{
    if( !isInFrameRange( frame ))
        return false;

    const float time  = _getTimeRange().x() + getDt() * frame;
    return load( time );
}

bool EventSource::load( const float time )
{
    if( time == _impl->currentTime )
        return true;

    const ssize_t updatedEvents = _load( time );
    if( updatedEvents < 0 )
    {
        LBERROR << "Timestamp " << time << "ms not loaded, no data or events"
                << std::endl;
        return false;
    }

    LBINFO << "Timestamp " << time << "ms loaded, updated " << updatedEvents
           << " event(s)" << std::endl;

    _impl->currentTime = time;
    return true;
}

Vector2ui EventSource::getFrameRange() const
{
    const Vector2f& interval = _getTimeRange();
    switch( _getType( ))
    {
    case SOURCE_EVENT:
        if( _hasEnded( ))
        {
            if( interval.x() == interval.y() && _impl->events.empty( ))
                // Do not return (0, 1) for empty sources.
                return Vector2ui( 0, 0 );
            return Vector2ui( std::floor( interval.x() / getDt( )),
                              std::floor( interval.y() / getDt() + 1 ));
        }
        else
            // Return only full frames [t, t+dt)
            return Vector2ui( std::floor( interval.x() / getDt( )),
                              std::floor( interval.y() / getDt( )));
    case SOURCE_FRAME:
    default:
        return Vector2ui( std::floor( interval.x() / getDt( )),
                          std::ceil( interval.y() / getDt( )));
    }
}

bool EventSource::isInFrameRange( uint32_t frame )
{
    const Vector2ui& frameRange = getFrameRange();
    return frame >= frameRange[0] && frame < frameRange[1];
}

float EventSource::getDt() const
{
    return _impl->dt;
}

void EventSource::setDt( const float dt )
{
    _impl->dt = dt;
}

}
