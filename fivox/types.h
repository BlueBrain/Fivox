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
    UNKNOWN,      //!< Unknown URI scheme
    COMPARTMENTS, //!< BBP compartment simulation reports
    SOMAS,        //!< BBP soma simulation reports
    SPIKES,       //!< BBP spike simulation reports
    SYNAPSES,     //!< BBP synapse positions
    VSD,          //!< BBP voltage sensitive dye simulation reports
};

/** Supported functor types */
enum FunctorType
{
    FUNCTOR_UNKNOWN,
    FUNCTOR_DENSITY,
    FUNCTOR_FIELD,
    FUNCTOR_FREQUENCY
};

}

// ITK forward decls
namespace itk
{
template< typename, unsigned > class Image;
template< typename > class SmartPointer;
}

#endif
