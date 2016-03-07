/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
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

#include <livre/core/data/DataSourcePlugin.h>

namespace fivox
{

/**
* Generate volume data sampling BBP simulation reports.
*/
class DataSource : public livre::DataSourcePlugin
{
public:
    explicit DataSource( const livre::DataSourcePluginData& data );
    virtual ~DataSource();

    /**
     * @param node is the LODNode for block information.
     * @return the data related to the node. If there is no data an
     * empty MemoryPtr is returned.
     */
    livre::MemoryUnitPtr getData( const livre::LODNode& node ) override;

    /**
     * Check whether the plugin supports the given URI and read mode.
     * @param data includes the URI and read mode.
     * @return true if the URI and read mode are handled.
     */
    static bool handles( const livre::DataSourcePluginData& data );

    /**
     * Updates the data source. Fivox data sources can be data streams,
     * which means that the number of frames changes over time.
     * This function updates the data source to the latest data received.
     */
    void update() final;

private:
    class Impl;
    std::unique_ptr< Impl > _impl;
    livre::LODNode internalNodeToLODNode( const livre::NodeId& internalNode ) const final;
};

}

#endif
