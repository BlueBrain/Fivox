// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_TYPES_H
#define FIVOX_TYPES_H

#include <vmmlib/vector.hpp>
#include <vector>

namespace fivox
{
struct Event;
typedef std::vector< Event > Events;
typedef vmml::vector< 3, float > Vector3f;
} // end namespace fivox

#endif
