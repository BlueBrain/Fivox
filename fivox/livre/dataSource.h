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
/**
* Generate volume data sampling BBPSDK simulation reports.
*/
class DataSource : public ::livre::VolumeDataSourcePlugin
{
public:
    explicit DataSource( const ::livre::VolumeDataSourcePluginData& data );
    virtual ~DataSource();

    ::livre::MemoryUnitPtr getData( const ::livre::LODNode& node ) override;

    static bool handles( const ::livre::VolumeDataSourcePluginData& data );

private:
    class Impl;
    std::unique_ptr< Impl > _impl;

    void internalNodeToLODNode( const ::livre::NodeId internalNode,
                                ::livre::LODNode& lodNode ) const final;
};

}
}

#endif
