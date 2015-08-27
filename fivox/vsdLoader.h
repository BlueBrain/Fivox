/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 *                     Jafet.VillafrancaDiaz@epfl.ch
 */
#ifndef FIVOX_VSDLOADER_H
#define FIVOX_VSDLOADER_H

#include <fivox/attenuationCurve.h>
#include <fivox/eventSource.h> // base class
#include <fivox/uriHandler.h>

namespace fivox
{
/**
 * Loads report data for voltage sensitive dye sampling.
 *
 * Expects an 'area' full compartment report with a single timestep containing
 * the surface areas of each compartment, and a 'voltages' soma report
 * containing the soma voltages for the given timestep.
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
    * Load the data at the given timestamp
    * @param time the timestamp of interest
    * @return true if time stamp can be retrieved from data source
    */
    bool load( float time ) final;

    /**
    * Load a new frame, based on the duration defined for each timestep (dt)
    * @param frame The frame number to be loaded
    * @return true if frame can be retrieved from data source
    */
    bool load( uint32_t frame ) final;

    /**
    * Set the attenuation curve that will be applied to the computed events
    * @param curve The attenuation curve to apply
    */
    void setCurve( const AttenuationCurve& curve ) final;

    /**
     * Gets the valid frame range according to data. The valid frames are in the
     * [a, b) range
     * @return the valid frame range
     */
    Vector2ui getFrameRange() final;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
