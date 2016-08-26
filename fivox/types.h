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

#include <brion/types.h>
#include <vmmlib/aabb.hpp>
#include <vmmlib/types.hpp>
#include <vmmlib/vector.hpp>
#include <memory>
#include <vector>

// ITK forward decls
namespace itk
{
template< typename, unsigned > class Image;
template< typename > class SmartPointer;
}

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
template< class TImage > class EventFunctor;
template< typename TImage > class ImageSource;

typedef std::shared_ptr< EventSource > EventSourcePtr;
typedef std::shared_ptr< const EventSource > ConstEventSourcePtr;

typedef itk::Image< uint8_t, 3 > ByteVolume;
typedef itk::Image< float, 3 > FloatVolume;
typedef std::shared_ptr< EventFunctor< ByteVolume >> ByteFunctorPtr;
typedef std::shared_ptr< EventFunctor< FloatVolume >> FloatFunctorPtr;

struct EventsDeleter
{
    void operator()( float* events ){ free( events ); }
};
typedef std::unique_ptr< float, EventsDeleter > Events;
typedef brion::floats EventValues;

using vmml::Vector2f;
using vmml::Vector3f;
using vmml::Vector2ui;
using vmml::Vector3ui;
using vmml::AABBf;

using servus::URI;

/** Supported data sources */
enum VolumeType
{
    TYPE_UNKNOWN,      //!< Unknown URI scheme
    TYPE_TEST,         /*!< Test type that creates fixed events
                            (e.g. for validation of different functors */
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
    FUNCTOR_LFP,     //!< LFP computation
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

#endif
