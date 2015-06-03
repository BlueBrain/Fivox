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

#include "dataSource.h"

#include <fivox/compartmentLoader.h>
#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <fivox/somaLoader.h>
#include <fivox/spikeLoader.h>
#include <fivox/vsdLoader.h>

#include <livre/core/Data/LODNode.h>
#include <livre/core/Data/MemoryUnit.h>
#include <livre/core/version.h>

#include <BBP/BBP.h>
#include <H5Cpp.h>

#include <boost/algorithm/string.hpp>
#include <boost/make_shared.hpp>
#include <boost/regex.hpp>

#ifdef FIVOX_USE_BBPTESTDATA
#  include <BBP/TestDatasets.h>
#endif

extern "C" int LunchboxPluginGetVersion() { return LIVRE_VERSION_ABI; }
extern "C" bool LunchboxPluginRegister()
{
    lunchbox::PluginRegisterer< fivox::livre::DataSource > registerer;
    return true;
}

namespace fivox
{
namespace livre
{
using boost::lexical_cast;

namespace
{
const size_t _defaultMaxBlockByteSize = LB_16MB;
const float _defaultCutoffDistance = 50.0f;
const float _defaultVoxelsPerUM = 1.0f;

typedef itk::Image< uint8_t, 3 > Image;
typedef Image::Pointer ImagePtr;
typedef ::fivox::EventFunctor< Image > Functor;
typedef ::fivox::ImageSource< Image, Functor > Source;
typedef Source::Pointer SourcePtr;
}

class DataSourceImpl
{
public:
    DataSourceImpl( const ::livre::VolumeDataSourcePluginData& pluginData )
        : source( Source::New( ))
    {
        const lunchbox::URI& uri = pluginData.getURI();
        std::string config = uri.getPath();
        std::string target = uri.getFragment();
        const bool useSpikes = (uri.getScheme() == "fivoxspikes");
        const bool useSoma = (uri.getScheme() == "fivoxsoma");
        const bool useVSD = (uri.getScheme() == "fivoxvsd");

#ifdef FIVOX_USE_BBPTESTDATA
        if( config.empty() )
        {
            if( useVSD )
                config = lunchbox::getExecutablePath() +
                    "/../share/Fivox/configs/BlueConfigVSD";
            else
                config = bbp::test::getBlueconfig();
            LBINFO << "Using test data " << config << std::endl;
        }

        if( target.empty( ))
        {
            target = useVSD ? "MiniColumn_0" : "L5CSPC";
            LBINFO << "Using target " << target << std::endl;
        }
#endif
        lunchbox::URI::ConstKVIter i = uri.findQuery( "dt" );
        const float dt = i == uri.queryEnd() ?
                                      0.1f : lexical_cast< float >( i->second );

        i = uri.findQuery( "report" );
        const std::string report = i == uri.queryEnd() ? "" : i->second;

        ::fivox::EventSourcePtr loader;
        if( useSpikes )
        {
            i = uri.findQuery( "duration" );
            const float duration = ( i == uri.queryEnd( )) ?
                                      10.f : lexical_cast< float >( i->second );

            i = uri.findQuery( "spikes" );
            std::string spikes;
            if( i != uri.queryEnd( ))
                spikes = i->second;

            loader = boost::make_shared< ::fivox::SpikeLoader >( config, target,
                                                                 spikes, dt,
                                                                 duration );
        }
        else if( useSoma )
            loader = boost::make_shared< ::fivox::SomaLoader >( config, target,
                                                                report, dt );
        else if( useVSD )
            loader = boost::make_shared< ::fivox::VSDLoader >( config, target,
                                                               dt );
        else
            loader = boost::make_shared< ::fivox::CompartmentLoader >(
                                                   config, target, report, dt );

        source->GetFunctor().setSource( loader );
        source->GetFunctor().setCutOffDistance( _defaultCutoffDistance );
#ifdef LIVRE_DEBUG_RENDERING
        std::cout << "Global space: " <<  loader->getBoundingBox() << std::endl;
#endif
    }

    ::livre::MemoryUnitPtr sample( const ::livre::LODNode& node,
                                   const ::livre::VolumeInformation& info )
        const
    {
        // Alloc voxels
        const vmml::Vector3i& voxels = info.maximumBlockSize;
        Image::SizeType vSize;
        vSize[0] = voxels[0];
        vSize[1] = voxels[1];
        vSize[2] = voxels[2];

        Image::RegionType region;
        region.SetSize( vSize );

        // Real-world coordinate setup
        ::fivox::EventSourcePtr loader = source->GetFunctor().getSource();
        const uint32_t frame = node.getNodeId().getFrame();
        loader->load( frame );

        const ::fivox::AABBf& bbox = loader->getBoundingBox();

        const Vector3f& baseSpacing = ( bbox.getDimension() + _borders )
                                      / info.voxels;
        const int32_t levelFromBottom = info.rootNode.getDepth() - 1 -
                                        node.getRefLevel();
        const float spacingFactor = 1 << levelFromBottom;

        Image::SpacingType spacing;
        spacing[0] = baseSpacing.find_max() * spacingFactor;
        spacing[1] = spacing[0];
        spacing[2] = spacing[0];

        const Vector3f& offset =
            ( bbox.getMin() - _borders / 2.0f ) + node.getRelativePosition() *
            Vector3f( bbox.getDimension() + _borders );

        Image::PointType origin;
        origin[0] = offset[0];
        origin[1] = offset[1];
        origin[2] = offset[2];

        // called from multiple render threads, only have one update running
        lunchbox::ScopedWrite mutex( _lock );
        ImagePtr image = source->GetOutput();
        image->SetRegions( region );
        image->SetSpacing( spacing );
        image->SetOrigin( origin );

#ifdef LIVRE_DEBUG_RENDERING
        std::cout << "Sample " << node.getRefLevel() << ' '
                  << node.getRelativePosition() << " (" << spacing << " @ "
                  << origin << 'x'
                  << baseSpacing * spacingFactor * voxels << ")"
                  << std::endl;
#endif
        source->Modified();
        source->Update();

        ::livre::AllocMemoryUnitPtr memoryUnit( new ::livre::AllocMemoryUnit );
        const size_t size = voxels[0] * voxels[1] * voxels[2] *
                            info.compCount * info.getBytesPerVoxel();
        memoryUnit->allocAndSetData( image->GetBufferPointer(), size );
        return memoryUnit;
    }

    SourcePtr source;
    Vector3f _borders;

private:
    mutable lunchbox::Lock _lock;
};

DataSource::DataSource( const ::livre::VolumeDataSourcePluginData& pluginData )
    : _impl( new DataSourceImpl( pluginData ))
{
    const lunchbox::URI& uri = pluginData.getURI();
    lunchbox::URI::ConstKVIter i = uri.findQuery( "voxelsPerUM" );
    lunchbox::URI::ConstKVIter j = uri.findQuery( "maxBlockSize" );

    const float voxelsPerUM = ( i == uri.queryEnd( )) ? _defaultVoxelsPerUM :
                                  lexical_cast< float >( i->second );

    const size_t maxBlockByteSize =
        ( j == uri.queryEnd( )) ? _defaultMaxBlockByteSize
                                : lexical_cast< size_t >( j->second );

    ::fivox::ConstEventSourcePtr loader =
          _impl->source->GetFunctor().getSource();
    const ::fivox::AABBf& bbox = loader->getBoundingBox();
    uint32_t depth = 0;
    const Vector3f totalTreeExactSize =
        ( bbox.getDimension() + _defaultCutoffDistance * 2.0f ) * voxelsPerUM;

    Vector3f blockExactDim = totalTreeExactSize;

    while (( ceil( blockExactDim.x( )) * ceil( blockExactDim.y( )) *
             ceil( blockExactDim.z( ))) > maxBlockByteSize )
    {
        blockExactDim = blockExactDim / 2.0f;
        depth++;
    }

    const size_t treeQuotient = 1 << depth;
    const vmml::Vector3ui blockDim( std::ceil( blockExactDim.x( )),
                                    std::ceil( blockExactDim.y( )),
                                    std::ceil( blockExactDim.z( )));

    const vmml::Vector3ui totalTreeSize = blockDim * treeQuotient;
    _impl->_borders = ( totalTreeSize / voxelsPerUM ) - bbox.getDimension();

    _volumeInfo.voxels = totalTreeSize;
    _volumeInfo.maximumBlockSize = blockDim;

    if( !::livre::fillRegularVolumeInfo( _volumeInfo ))
       LBTHROW( std::runtime_error( "Cannot setup the regular tree" ));

    // SDK uses microns, volume information uses meters
    _volumeInfo.boundingBox = bbox / 1000000.f;
}

DataSource::~DataSource()
{
    delete _impl;
}

::livre::MemoryUnitPtr DataSource::getData( const ::livre::LODNode& node )
{
    try
    {
        return _impl->sample( node, getVolumeInformation( ));
    }
    catch( const std::exception& e )
    {
        LBWARN << "sample failed: " << e.what() << std::endl;
        return ::livre::MemoryUnitPtr();
    }
    catch( const H5::Exception& e )
    {
        LBWARN << "sample failed: " << e.getDetailMsg() << std::endl;
        return ::livre::MemoryUnitPtr();
    }
}

void DataSource::internalNodeToLODNode(
    const ::livre::NodeId internalNode, ::livre::LODNode& lodNode ) const
{
    const uint32_t refLevel = internalNode.getLevel();
    const vmml::Vector3ui& bricksInRefLevel =
                                  _volumeInfo.rootNode.getBlockSize( refLevel );
    const vmml::AABB< int32_t > localBlockPos( internalNode.getPosition(),
                                               internalNode.getPosition() + 1u);

    const uint32_t index = bricksInRefLevel.find_max_index( );
    const Vector3f boxCoordMin = Vector3f( localBlockPos.getMin( ))
                                 / bricksInRefLevel[index];
    const Vector3f boxCoordMax = Vector3f( localBlockPos.getMax( ))
                                 / bricksInRefLevel[index];

#ifdef LIVRE_DEBUG_RENDERING
    LBINFO << " Internal Node to LOD Node" << std::endl
           << " Node Id " << internalNode
           << " BricksInRefLevel " << bricksInRefLevel << std::endl
           << " lBoxCoordMin " << boxCoordMin << std::endl
           << " lBoxCoordMax " << boxCoordMax << std::endl
           << " volume world size " << _volumeInfo.worldSize << std::endl
           << std::endl;
#endif

    lodNode = ::livre::LODNode( internalNode,
                                _volumeInfo.maximumBlockSize -
                                _volumeInfo.overlap * 2,
                                AABBf( boxCoordMin * _volumeInfo.worldSize -
                                       _volumeInfo.worldSize * 0.5f,
                                       boxCoordMax * _volumeInfo.worldSize -
                                       _volumeInfo.worldSize * 0.5f ));
}

bool DataSource::handles( const ::livre::VolumeDataSourcePluginData& data )
{
    const std::string fivox = "fivox";
    const std::string& scheme = data.getURI().getScheme();
    return scheme.substr( 0, fivox.size( )) == fivox;
}

}
}
