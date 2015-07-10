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
Processes an URI and provides all the parameters specified in it.
@snippet apps/voxelize.cpp Usage
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
     * If no target is specified will return the given default target. If this
     * is not specified (and TestData is available), will return 'Column' when
     * using spikes or synapses, 'Layer1' otherwise.
     *
     * @return the specified target name.
     */
    std::string getTarget( const std::string& defaultTarget ) const;

    /**
     * Get the specified report name.
     *
     * If the given report is empty, returns 'soma' (for volume type soma) or
     * 'voltage' (for all others), or the default reports if test data circuit
     * is used.
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


    std::string getSpikes() const;
    /**
     * Get the specified duration.
     *
     * @return the specified duration. If invalid or empty, return getDt().
     */
    float getDuration() const;

    /**
     * Get the specified path to a dye curve file
     * @return the specified path to the dye curve file
     */
    std::string getDyeCurve() const;

    /**
     * Get the specified resolution (voxels per micrometer).
     *
     * @return the specified resolution. If invalid or empty, return 1
     */
    float getResolution() const;

    /**
     * Get the specified maximum size per block (bytes).
     *
     * @return the specified maximum size per block. If invalid or empty,
     * return 16MB
     */
    size_t getMaxBlockSize() const;

    /**
     * Get the type of the volume that is being loaded (present in the URI
     * schema)
     * @return the type of the volume
     */
    VolumeType getType() const;

    /** @return a new loader for the given parameters. */
    EventSourcePtr newLoader() const;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
};
}

#endif
