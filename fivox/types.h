/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 */
#ifndef FIVOX_TYPES_H
#define FIVOX_TYPES_H

#include <vmmlib/aabb.hpp>
#include <vmmlib/vector.hpp>
#include <memory>
#include <vector>

/** Field Voxelization Library */
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
    FUNCTOR_DENSITY,
    FUNCTOR_FIELD,
    FUNCTOR_FREQUENCY
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
typedef itk::Image< uint8_t, 3 > Volume;
typedef std::shared_ptr< fivox::EventFunctor< Volume >> FunctorPtr;
}

#endif
