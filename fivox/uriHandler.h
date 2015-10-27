/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
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

#ifndef FIVOX_URIHANDLER_H
#define FIVOX_URIHANDLER_H

#include <fivox/types.h>
#include <memory>

namespace fivox
{

/**
 * Process an URI to provide all the parameters specified in it.
 * @snippet apps/voxelize.cpp Usage
 */
class URIHandler
{
public:
    /**
     * Construct a new URI handler.
     *
     * @param parameters URI containing the parameters in the specified form
     */
    explicit URIHandler( const std::string& parameters );
    virtual ~URIHandler(); //!< Destruct this parameter processor

    /**
     * Get the specified BlueConfig file path.
     *
     * @return path to the specified BlueConfig. If empty, return the
     * TestData's BlueConfig if available
     */
    std::string getConfig() const;

    /**
     * Get the specified target name.
     *
     * @return If no target was given at construction return the given
     *         default target.
     *         If this is also empty (and TestData is available), return
     *         'Column' when using spikes or synapses, 'Layer1' otherwise.
     */
    std::string getTarget( const std::string& defaultTarget ) const;
    std::string getTarget() const;

    /**
     * Get the specified report name.
     *
     * If the given report is empty, returns 'soma' (for soma and vsd loaders)
     * or 'voltage' (for all others), or the default reports if test data
     * circuit is used.
     *
     * @return the specified report name.
     */
    std::string getReport() const;

    /**
     * Get the specified timestep (dt).
     *
     * @return the specified timestep. If invalid or empty, return 10.
     */
    float getDt() const;

    /** @return URI to spikes source, empty by default */
    std::string getSpikes() const;

    /**
     * Get the specified duration.
     *
     * @return the specified duration. If invalid or empty, return getDt().
     */
    float getDuration() const;

    /**
     * Get the magnitude value which is mulitplied on each sampled voxel value.
     * The default magnitude depends on VolumeType:
     * - 0.1 for COMPARTMENTS and SOMAS
     * - 1.5 / getDuration() for SPIKES
     * - 1.0 for SYNAPSES and VSD
     *
     * @return magnitude for each voxel (multiplied on its sampled value)
     */
    float getMagnitude() const;

    /**
     * Get the specified path to a dye curve file
     * @return the specified path to the dye curve file
     */
    std::string getDyeCurve() const;

    /**
     * Get the specified resolution in voxels per unit (typically um).
     *
     * @return the specified resolution. If invalid or empty, return 10.
     */
    float getResolution() const;

    /**
     * Get the specified maximum size per block (bytes).
     *
     * @return the specified maximum size per block. If invalid or empty, return
     *         64MB
     */
    size_t getMaxBlockSize() const;

    /**
     * Get the type of the volume that is being loaded (present in the URI
     * schema)
     * @return the type of the volume
     */
    VolumeType getType() const;

    /**
     * Available functors are "density", "field" and "frequency". If "functor"
     * is unspecified, the default functor for the VolumeType is returned:
     * - FUNCTOR_DENSITY for SYNAPSES
     * - FUNCTOR_FREQUENCY for SPIKES
     * - FUNCTOR_FIELD for COMPARTMENTS, SOMAS and VSD
     *
     * @return the type of the functor to use, use VolumeType default functor
     *          if unspecified.
     */
    FunctorType getFunctorType() const;

    /** @return a new image source for the given parameters and pixel type. */
    template< class T >
    itk::SmartPointer< ImageSource< itk::Image< T, 3 >>> newImageSource() const;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
