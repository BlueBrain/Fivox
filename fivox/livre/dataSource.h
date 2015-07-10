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

#ifndef FIVOX_LIVRE_DATASOURCE_H
#define FIVOX_LIVRE_DATASOURCE_H

#include <livre/core/Data/VolumeDataSourcePlugin.h>

namespace fivox
{
namespace livre
{
class DataSourceImpl;

/*
//! [Usage]
Generate volume data sampling BBPSDK simulation reports.

URIs must be in the forms:
- Compartment reports:
    fivox[compartment]://BlueConfig?dt=float,report=string#target
- Soma reports:
    fivoxsoma://BlueConfig?dt=float,report=string#target
- Spike reports:
    fivoxspikes://BlueConfig?dt=float,duration=float,spikes=path#target
- Voltage-sensitive dye reports:
    fivoxvsd://BlueConfig?dt=float,dyecurve=string#target

Parameters:
- BlueConfig: BlueConfig file path
              (default: 'configs/BlueConfigVSD' for VSD reports,
              BBPTestData otherwise)
- dt: duration in milliseconds of the timestep (default: dt of the report)
- duration: size of the time window to load spikes (default: 10)
- target: name of the BlueConfig target (default: the circuit target)
- report: name of the compartment report (default: 'voltage' for compartment
          reports, 'soma' for soma reports)
- spikes: path to an alternate out.dat/out.spikes file
          (default: SpikesPath specified in the BlueConfig)
- dyecurve: path to the dye curve file to apply, e.g. attenuation
            (only on VSD reports)
- voxelsPerUM: number of voxels per micrometer (default: 1.0)
- maxBlockSize: maximum memory usage allowed for one block in bytes
                (default: 16777216 bytes -- 16MB)
//! [Usage]
*/
class DataSource : public ::livre::VolumeDataSourcePlugin
{
public:
    DataSource( const ::livre::VolumeDataSourcePluginData& pluginData );
    virtual ~DataSource();

    ::livre::MemoryUnitPtr getData( const ::livre::LODNode& node ) override;

    static bool handles( const ::livre::VolumeDataSourcePluginData& data );

private:
    void internalNodeToLODNode( const ::livre::NodeId internalNode,
                                ::livre::LODNode& lodNode ) const final;

    DataSourceImpl* const _impl;

};

}
}

#endif
