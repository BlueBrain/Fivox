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

template< typename TImage >
using ImageSourcePtr = itk::SmartPointer< ImageSource< TImage >>;

template< typename TImage >
using EventFunctorPtr = std::shared_ptr< EventFunctor< TImage >>;

/** Supported data sources */
enum class VolumeType
{
    unknown,      //!< Unknown URI scheme
    generic,      /*!< Generic type that loads events from file (if present),
                       or creates fixed events (e.g. for validation of different
                       functors) */
    compartments, //!< BBP compartment simulation reports
    somas,        //!< BBP soma simulation reports
    spikes,       //!< BBP spike simulation reports
    synapses,     //!< BBP synapse positions
    vsd,          //!< BBP voltage sensitive dye simulation reports
};

/** Supported functor types */
enum class FunctorType
{
    unknown,
    density,  //!< sum( magnitude of events in voxel ) / volume of voxel
    lfp,      //!< LFP computation
    field,    //!< quadratic falloff of magnitude in space
    frequency //!< maximum magnitude of all events in voxel
};

/** @internal Different types of event sources which defines
    EventSource::getFrameRange */
enum class SourceType
{
    event, //!< e.g. spikes reports
    frame  //!< e.g. compartment reports
};

/** Indicates to consider all data for potential rescaling. */
const Vector2f FULLDATARANGE( -std::numeric_limits< float >::infinity(),
                               std::numeric_limits< float >::infinity( ));

}

#endif
