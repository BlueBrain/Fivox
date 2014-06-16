// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_EVENT_H
#define FIVOX_EVENT_H

#include <fivox/defines.h>
#include <fivox/types.h>

namespace fivox
{
/** A positional event with a value to be sampled. */
struct Event
{
  Event( const Vector3f& pos, const float val )
      : position( pos )
      , value( val )
  {}

  Vector3f position;
  float value;
};
} // end namespace fivox

#endif
