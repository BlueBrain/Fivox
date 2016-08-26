
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
    enum EventOffsets
    {
        POSX = 0,
        POSY,
        POSZ,
        RADIUS,
        VALUE,
        NUM_OFFSETS
    };

    explicit Impl( const URIHandler& params )
        : dt( params.getDt( ))
        , currentTime( -1.f )
        , cutOffDistance( params.getCutoffDistance( ))
        , alignBoundary( 32 )
        , numEvents( 0 )
        , allocSize( 0 )
    {}

    void resize( const size_t numEvents_ )
    {
        numEvents = numEvents_;
        if( numEvents_ < allocSize )
            return;

        allocSize = numEvents_;
        const size_t size = numEvents * EventOffsets::NUM_OFFSETS;
        void* ptr;
        if( posix_memalign( &ptr, alignBoundary, size * sizeof(float) ))
        {
            LBWARN << "Memory alignment failed. "
                   << "Trying normal allocation" << std::endl;
            ptr = calloc( size, sizeof(float) );
            if( !ptr )
                LBTHROW( std::bad_alloc( ));
        }
        events.reset((float*) ptr );
    }

    const float* getPositionsX() const
    {
        return events.get() + numEvents * EventOffsets::POSX;
    }

    const float* getPositionsY() const
    {
        return events.get() + numEvents * EventOffsets::POSY;
    }

    const float* getPositionsZ() const
    {
        return events.get() + numEvents * EventOffsets::POSZ;
    }

    const float* getRadii() const
    {
        return events.get() + numEvents * EventOffsets::RADIUS;
    }

    const float* getValues() const
    {
        return events.get() + numEvents * EventOffsets::VALUE;
    }

    float dt;
    float currentTime;
    const float cutOffDistance;

    const size_t alignBoundary;
    size_t numEvents;
    size_t allocSize;
    Events events;
    AABBf boundingBox;

#ifdef USE_BOOST_GEOMETRY
    typedef bgi::rtree< Value, bgi::rstar< maxElemInNode, minElemInNode > > RTree;
    RTree rtree;

    void buildRTree()
    {
        if( !rtree.empty( ))
            return;

        LBINFO << "Building rtree for " << numEvents << " events"
               << std::endl;
        Values positions;
        positions.reserve( numEvents );

        for( size_t i = 0; i < numEvents; i++ )
        {
            const Point point( getPositionsX()[i],
                               getPositionsY()[i],
                               getPositionsZ()[i] );
            positions.push_back( std::make_pair( point, i ));
        }

        RTree rt( positions.begin(), positions.end( ));
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

float& EventSource::operator[]( const size_t index )
{
    return _impl->events.get()[ _impl->numEvents * Impl::EventOffsets::VALUE +
                                index ];
}

size_t EventSource::getNumEvents() const
{
    return _impl->numEvents;
}

const float* EventSource::getPositionsX() const
{
    return _impl->getPositionsX();
}

const float* EventSource::getPositionsY() const
{
    return _impl->getPositionsY();
}

const float* EventSource::getPositionsZ() const
{
    return _impl->getPositionsZ();
}

const float* EventSource::getRadii() const
{
    return _impl->getRadii();
}

const float* EventSource::getValues() const
{
    return _impl->getValues();
}

EventValues EventSource::findEvents( const AABBf& area LB_UNUSED ) const
{
    EventValues eventValues;
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

        eventValues.reserve( hits.size( ));
        for( const Value& value : hits )
            eventValues.push_back( getValues()[value.second] );
    }
    else
#endif
    // return empty
    {
        static bool first = true;
        if( first )
        {
            LBWARN << "RTree not available for findEvents. "
                   << "No events will be returned" << std::endl;
            first = false;
        }
    }
    return eventValues;
}

const AABBf& EventSource::getBoundingBox() const
{
    return _impl->boundingBox;
}

float EventSource::getCutOffDistance() const
{
    return _impl->cutOffDistance;
}

void EventSource::resize( const size_t size )
{
    _impl->resize( size );
}

void EventSource::update( const size_t i, const Vector3f& pos,
                          const float rad, const float val )
{
    const size_t size( getNumEvents( ));
    if( size <= i )
    {
        LBWARN << "The specified index is not valid. Event not added"
               << std::endl;
        return;
    }

    _impl->boundingBox.merge( pos );
    _impl->events.get()[ i + size * Impl::EventOffsets::POSX ] = pos[0];
    _impl->events.get()[ i + size * Impl::EventOffsets::POSY ] = pos[1];
    _impl->events.get()[ i + size * Impl::EventOffsets::POSZ ] = pos[2];

    // radius is inverted to improve performance at computing time
    // e.g. LFP functor
    if( std::abs( rad ) > std::numeric_limits< float >::epsilon( )) // rad != 0
        _impl->events.get()[i + size * Impl::EventOffsets::RADIUS] =  1.f / rad;

    _impl->events.get()[ i + size * Impl::EventOffsets::VALUE ] = val;

#ifdef USE_BOOST_GEOMETRY
    _impl->rtree.clear();
#endif
}

void EventSource::buildRTree()
{
#ifdef USE_BOOST_GEOMETRY
    _impl->buildRTree();
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
            if( interval.x() == interval.y() && _impl->numEvents == 0 )
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
