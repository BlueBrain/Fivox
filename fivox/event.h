
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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

#ifndef FIVOX_EVENT_H
#define FIVOX_EVENT_H

#include <fivox/defines.h>
#include <fivox/types.h>

namespace fivox
{
/** A positional event with a value to be sampled and optionally a radius of
 *  influence (0 by default)
*/
struct Event
{
  Event( const Vector3f& pos, const float val, const float rad = 0.f )
      : position( pos )
      , value( val )
      , radius( rad )
  {}

  const Vector3f position;
  float value;
  float radius;
};
} // end namespace fivox

#endif
