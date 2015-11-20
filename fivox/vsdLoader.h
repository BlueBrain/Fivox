/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */
#ifndef FIVOX_VSDLOADER_H
#define FIVOX_VSDLOADER_H

#include <fivox/attenuationCurve.h>
#include <fivox/eventSource.h> // base class

namespace fivox
{
/**
 * Loads report data for voltage sensitive dye sampling.
 *
 * Expects an 'area' full compartment report with a single timestep containing
 * the surface areas of each compartment, and a soma report containing the soma
 * voltages for the given timestep.
 */
class VSDLoader : public EventSource
{
public:
    /**
    * Construct a new vsd event source.
    *
    * @param params the URIHandler object containing the parameters
    * to define the event source
    * @throw H5::exception or std::exception on error
    */
    explicit VSDLoader( const URIHandler& params );
    virtual ~VSDLoader(); //!< Destruct this vsd event source

    /**
    * Set the attenuation curve that will be applied to the computed events
    * @param curve The attenuation curve to apply
    */
    void setCurve( const AttenuationCurve& curve ) final;

private:
    /** @name Abstract interface implementation */
    //@{
    Vector2f _getTimeRange() const final;
    bool _load( float time ) final;
    SourceType _getType() const final { return SOURCE_FRAME; }
    //@}

    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
