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

#include "dataSource.h"

#include <fivox/fieldFunctor.h>
#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <fivox/synapseLoader.h>
#include <fivox/uriHandler.h>

#include <livre/core/data/LODNode.h>
#include <livre/core/data/MemoryUnit.h>
#include <livre/core/version.h>

#include <lunchbox/scopedMutex.h>
#include <lunchbox/pluginRegisterer.h>

extern "C" int LunchboxPluginGetVersion() { return LIVRECORE_VERSION_ABI; }
extern "C" bool LunchboxPluginRegister()
{
    lunchbox::PluginRegisterer< fivox::DataSource > registerer;
    return true;
}

namespace fivox
{
namespace
{
typedef ByteVolume::Pointer VolumePtr;

typedef ::fivox::ImageSource< ByteVolume > ImageSource;
typedef ImageSource::Pointer ImageSourcePtr;

typedef typename ImageSource::FunctorPtr FunctorPtr;
typedef fivox::FieldFunctor< ByteVolume > VolumeFieldFunctor;
typedef std::shared_ptr< VolumeFieldFunctor > FieldFunctorPtr;
}

class DataSource::Impl
{
public:
    explicit Impl( const livre::VolumeDataSourcePluginData& pluginData )
        : params( std::to_string( pluginData.getURI( )))
        , source( params.newImageSource< uint8_t >( ))
    {
    }

    livre::MemoryUnitPtr sample( const livre::LODNode& node,
                                 const livre::VolumeInformation& info ) const
    {
        ::fivox::EventSourcePtr loader = source->getFunctor()->getSource();
        const uint32_t frame = node.getNodeId().getFrame();
        if( !loader->load( frame ))
            return livre::MemoryUnitPtr();

        // Alloc voxels
        const vmml::Vector3i& voxels = info.maximumBlockSize;
        ByteVolume::SizeType vSize;
        vSize[0] = voxels[0];
        vSize[1] = voxels[1];
        vSize[2] = voxels[2];

        ByteVolume::RegionType region;
        region.SetSize( vSize );

        // Real-world coordinate setup
        const ::fivox::AABBf& bbox = loader->getBoundingBox();
        const Vector3f& baseSpacing = ( bbox.getSize() + _borders )
                                      / info.voxels;
        const int32_t levelFromBottom = info.rootNode.getDepth() - 1 -
                                        node.getRefLevel();
        const float spacingFactor = 1 << levelFromBottom;

        ByteVolume::SpacingType spacing;
        spacing[0] = baseSpacing.find_max() * spacingFactor;
        spacing[1] = spacing[0];
        spacing[2] = spacing[0];

        const Vector3f& offset =
            ( bbox.getMin() - _borders / 2.0f ) + node.getRelativePosition() *
            Vector3f( bbox.getSize() + _borders );

        ByteVolume::PointType origin;
        origin[0] = offset[0];
        origin[1] = offset[1];
        origin[2] = offset[2];

        // called from multiple render threads, only have one update running
        lunchbox::ScopedWrite mutex( _lock );
        VolumePtr output = source->GetOutput();
        output->SetRegions( region );
        output->SetSpacing( spacing );
        output->SetOrigin( origin );

#ifdef LIVRE_DEBUG_RENDERING
        std::cout << "Sample " << node.getRefLevel() << ' '
                  << node.getRelativePosition() << " (" << spacing << " @ "
                  << origin << 'x'
                  << baseSpacing * spacingFactor * voxels << ")"
                  << std::endl;
#endif
        source->Modified();
        source->Update();

        livre::AllocMemoryUnitPtr memoryUnit( new livre::AllocMemoryUnit );
        const size_t size = voxels[0] * voxels[1] * voxels[2] *
                            info.compCount * info.getBytesPerVoxel();
        memoryUnit->allocAndSetData( output->GetBufferPointer(), size );
        return memoryUnit;
    }

    void update( livre::VolumeInformation& info )
    {
        ::fivox::EventSourcePtr loader = source->getFunctor()->getSource();
        const Vector2ui& frameRange = loader->getFrameRange();
        if( frameRange[1] > 0 ) // is any frame present
            info.frameRange = frameRange;
    }

    const URIHandler params;
    ImageSourcePtr source;
    Vector3f _borders;

private:
    mutable lunchbox::Lock _lock;
};

DataSource::DataSource( const livre::VolumeDataSourcePluginData& pluginData )
    : _impl( new DataSource::Impl( pluginData ))
{
    const float resolution = _impl->params.getResolution();
    const size_t maxBlockByteSize = _impl->params.getMaxBlockSize( );

    FunctorPtr functor = _impl->source->getFunctor();
    ::fivox::EventSourcePtr loader = functor->getSource();

    const ::fivox::AABBf& bbox = loader->getBoundingBox();
    uint32_t depth = 0;
    const Vector3f fullResolution =
        ( bbox.getSize() + std::sqrt( 1.0 / _impl->params.getMaxError()) * 2.0f ) * resolution;
    Vector3f blockResolution = fullResolution;

    // maxTextureSize value should be retrieved from OpenGL. But at this
    // point in time there may be no GL context. So a general object is
    // needed in Livre to query the OpenGL device properties.
    const size_t maxTextureSize = 2048;
    while( true )
    {
        if( blockResolution.product() < maxBlockByteSize &&
            blockResolution.x() < maxTextureSize &&
            blockResolution.y() < maxTextureSize &&
            blockResolution.z() < maxTextureSize )
        {
            break;
        }
        blockResolution = blockResolution / 2.0f;
        ++depth;
    }

    vmml::Vector3ui blockDim( std::ceil( blockResolution.x( )),
                              std::ceil( blockResolution.y( )),
                              std::ceil( blockResolution.z( )));
    if( blockDim.x() > 8 )
        blockDim.x() -= (blockDim.x() % 8);
    if( blockDim.y() > 8 )
        blockDim.y() -= (blockDim.y() % 8);
    if( blockDim.z() > 8 )
        blockDim.z() -= (blockDim.z() % 8);

    const size_t treeQuotient = 1 << depth;
    const vmml::Vector3ui totalTreeSize = blockDim * treeQuotient;
    _impl->_borders = vmml::Vector3f( totalTreeSize ) / resolution -
                      bbox.getSize();

    _volumeInfo.voxels = totalTreeSize;
    _volumeInfo.maximumBlockSize = blockDim;

    if( !livre::fillRegularVolumeInfo( _volumeInfo ))
        LBTHROW( std::runtime_error( "Cannot setup the regular tree" ));

    // SDK uses microns, volume information uses meters
    _volumeInfo.boundingBox = ::fivox::AABBf( bbox.getMin() / 1000000.f,
                                              bbox.getMax() / 1000000.f );
}

DataSource::~DataSource()
{}

livre::MemoryUnitPtr DataSource::getData( const livre::LODNode& node )
{
    try
    {
        return _impl->sample( node, _volumeInfo );
    }
    catch( const std::exception& e )
    {
        LBERROR << "sample failed: " << e.what() << std::endl;
        return livre::MemoryUnitPtr();
    }
    catch( ... )
    {
        LBERROR << "sample failed" << std::endl;
        return livre::MemoryUnitPtr();
    }
}

void DataSource::internalNodeToLODNode( const livre::NodeId& internalNode,
                                        livre::LODNode& lodNode ) const
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

    LBVERB << "Internal Node to LOD Node" << std::endl
           << "  node id " << internalNode << std::endl
           << "  bricks in reflevel " << bricksInRefLevel << std::endl
           << "  min bbox " << boxCoordMin << std::endl
           << "  max bbox " << boxCoordMax << std::endl
           << "  volume world size " << _volumeInfo.worldSize << std::endl
           << std::endl;

    lodNode = livre::LODNode( internalNode,
                                _volumeInfo.maximumBlockSize -
                                _volumeInfo.overlap * 2,
                                AABBf( boxCoordMin * _volumeInfo.worldSize -
                                       _volumeInfo.worldSize * 0.5f,
                                       boxCoordMax * _volumeInfo.worldSize -
                                       _volumeInfo.worldSize * 0.5f ));
}

bool DataSource::handles( const livre::VolumeDataSourcePluginData& data )
{
    const std::string fivox = "fivox";
    const std::string& scheme = data.getURI().getScheme();
    return scheme.substr( 0, fivox.size( )) == fivox;
}

void DataSource::update()
{
    _impl->update( _volumeInfo );
}

}
