// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "eventSource.h"
#include "event.h"

namespace fivox
{
namespace detail
{
class EventSource
{
public:

  Events events;
  AABBf boundingBox;
};
}

EventSource::EventSource()
    : _impl( new detail::EventSource )
{}

EventSource::~EventSource()
{
    delete _impl;
}

const Events& EventSource::getEvents() const
{
    return _impl->events;
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
    _impl->boundingBox.merge( event.position );
    _impl->events.push_back( event );
}

void EventSource::update( const size_t index, const float value )
{
    assert( index < _impl->events.size( ));
    _impl->events[ index ].value = value;
}

}
