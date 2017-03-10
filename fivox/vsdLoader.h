/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
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

#ifndef FIVOX_VSDLOADER_H
#define FIVOX_VSDLOADER_H

#include <fivox/api.h>
#include <fivox/attenuationCurve.h>
#include <fivox/eventSource.h> // base class

namespace fivox
{
/**
 * Loads BBP report data for voltage sensitive dye sampling.
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
    FIVOX_API explicit VSDLoader(const URIHandler& params);
    FIVOX_API virtual ~VSDLoader(); //!< Destruct this vsd event source

    /**
    * Set the attenuation curve that will be applied to the computed events
    *
    * @param curve The attenuation curve to apply
    */
    FIVOX_API void setCurve(const AttenuationCurve& curve);

    /**
     * Enable the interpolation of the attenuation values from the dye curve
     *
     * @param interpolate true to interpolate the attenuation values before
     * applying them; false to use the attenuation values as they are in the
     * dye curve file
     */
    FIVOX_API void setInterpolation(bool interpolate);

    /**
     * @return All GIDs loaded
     */
    FIVOX_API const brion::GIDSet& getGIDs() const;

    /**
     * @return The soma positions of all the cells loaded
     */
    FIVOX_API const brion::Vector3fs getSomaPositions() const;

    /**
    * Set the resting potential that will be used for the computation of the VSD
    *
    * @param millivolts resting potential in mV
    */
    FIVOX_API void setRestingPotential(float millivolts);

    /**
    * Set the multiplier for surface area in background fluorescence term value
    *
    * @param factor multiplier for surface area in background fluorescence term
    * value
    */
    FIVOX_API void setAreaMultiplier(float factor);

    /**
     * Enable the spike filtering
     * @param enable true to enable spike filtering; false to use the input
     * voltages as they are
     */
    FIVOX_API void setSpikeFilter(bool enable);

    /**
     * When the spike filtering is enabled, use this to set the action potential
     * threshold, in millivolts
     *
     * @param apThreshold action potential threshold, in millivolts
     */
    FIVOX_API void setApThreshold(float apThreshold);

private:
    /** @name Abstract interface implementation */
    //@{
    Vector2f _getTimeRange() const final;
    ssize_t _load(size_t chunkIndex, size_t numChunks) final;
    SourceType _getType() const final { return SourceType::frame; }
    size_t _getNumChunks() const final { return 1; }
    //@}

    class Impl;
    std::unique_ptr<Impl> _impl;
};
}

#endif
