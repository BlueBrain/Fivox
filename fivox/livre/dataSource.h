/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
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
namespace detail{ class DataSource; }

/**
 * Generate volume data sampling BBPSDK simulation reports.
 *
 * Used for volume URIs in the forms:
 * - Compartment reports: fivox[Compartment]://BlueConfig?time=float,report=string#target
 * - Soma reports: fivoxSoma://BlueConfig?time=float,report=string#target
 * - Spike reports: fivoxSpikes://BlueConfig?time=float,duration=float,spikes=path#target
 *
 * Parameters:
 * - time: initial timestep to load
 * - duration: size of the time window to load spikes
 * - target: name of the BlueConfig target
 * - report: name of the compartment report
 * - spikes: path to an alternate out.dat/out.spikes file
 */
class DataSource : public ::livre::VolumeDataSourcePlugin
{
public:
    DataSource( const ::livre::VolumeDataSourcePluginData& pluginData );
    virtual ~DataSource();

    ::livre::MemoryUnitPtr getData( const ::livre::LODNode& node ) override;

    static bool handles( const ::livre::VolumeDataSourcePluginData& data );

private:
    detail::DataSource* const _impl;

};

}
}

#endif
