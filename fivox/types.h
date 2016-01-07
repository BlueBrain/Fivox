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

#ifndef FIVOX_TYPES_H
#define FIVOX_TYPES_H

#include <vmmlib/aabb.hpp>
#include <vmmlib/vector.hpp>
#include <memory>
#include <vector>

/**
 * Field Voxelization Library
 *
 * An ImageSource implements an itk::ImageSource. It uses an EventFunctor to
 * sample Event into the configured volume. The events are loaded by an
 * EventSource.
 */
namespace fivox
{
class EventSource;
class URIHandler;
struct Event;
template< class TImage > class EventFunctor;
template< typename TImage > class ImageSource;

typedef std::shared_ptr< EventSource > EventSourcePtr;
typedef std::shared_ptr< const EventSource > ConstEventSourcePtr;

typedef std::vector< Event > Events;

using vmml::Vector2f;
using vmml::Vector3f;
using vmml::Vector2ui;
using vmml::AABBf;

/** Supported data sources */
enum VolumeType
{
    TYPE_UNKNOWN,      //!< Unknown URI scheme
    TYPE_COMPARTMENTS, //!< BBP compartment simulation reports
    TYPE_SOMAS,        //!< BBP soma simulation reports
    TYPE_SPIKES,       //!< BBP spike simulation reports
    TYPE_SYNAPSES,     //!< BBP synapse positions
    TYPE_VSD,          //!< BBP voltage sensitive dye simulation reports
};

/** Supported functor types */
enum FunctorType
{
    FUNCTOR_UNKNOWN,
    FUNCTOR_DENSITY, //!< sum( magnitude of events in voxel ) / volume of voxel
    FUNCTOR_FIELD,   //!< quadratic falloff of magnitude in space
    FUNCTOR_FREQUENCY //!< maximum magnitude of all events in voxel
};

/** @internal Different types of event sources which defines
    EventSource::getFrameRange */
enum SourceType
{
    SOURCE_EVENT, //!< e.g. spikes reports
    SOURCE_FRAME //!< e.g. compartment reports
};

}

// ITK forward decls
namespace itk
{
template< typename, unsigned > class Image;
template< typename > class SmartPointer;
}

namespace
{
typedef itk::Image< float, 3 > Volume;
typedef std::shared_ptr< fivox::EventFunctor< Volume >> FunctorPtr;
}

#endif
