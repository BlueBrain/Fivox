
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
const size_t _defaultTotalSize = 16384;
const size_t _defaultBlockSize = 256;

typedef itk::Image< uint8_t, 3 > Image;
typedef Image::Pointer ImagePtr;
typedef ::fivox::EventFunctor< Image > Functor;
typedef ::fivox::ImageSource< Image, Functor > Source;
typedef Source::Pointer SourcePtr;
}

namespace detail
{
class DataSource
{
public:
    DataSource( const ::livre::VolumeDataSourcePluginData& pluginData )
        : source( Source::New( ))
    {
        const lunchbox::URI& uri = pluginData.getURI();
        std::string config = uri.getPath();
        std::string target = uri.getFragment();
        const bool useSpikes = (uri.getScheme() == "fivoxSpikes");
        const bool useSoma = (uri.getScheme() == "fivoxSoma");
        const bool useVSD = (uri.getScheme() == "fivoxVSD");

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

        if( !useSpikes && target.empty( ))
        {
            target = useVSD ? "MiniColumn_0" : "L5CSPC";
            LBINFO << "Using target " << target << std::endl;
        }
#endif
        lunchbox::URI::ConstKVIter i = uri.findQuery( "time" );
        const float time = i == uri.queryEnd() ?
                                    0.f : lexical_cast< float >( i->second );

        ImagePtr output = source->GetOutput();
        ::fivox::EventSourcePtr loader;
        if( useSpikes )
        {
            i = uri.findQuery( "duration" );
            const float duration = i == uri.queryEnd() ?
                                      10.f : lexical_cast< float >( i->second );

            i = uri.findQuery( "spikes" );
            if( i != uri.queryEnd( ))
                target = i->second;

            loader = boost::make_shared< ::fivox::SpikeLoader >( config, target,
                                                                 time,
                                                                 duration );
        }
        else if( useSoma )
            loader = boost::make_shared< ::fivox::SomaLoader >( config, target,
                                                                time );
        else if( useVSD )
            loader = boost::make_shared< ::fivox::VSDLoader >( config, target,
                                                               time );
        else
            loader = boost::make_shared< ::fivox::CompartmentLoader >(
                                             config, target, time );

        source->GetFunctor().setSource( loader );
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
        ::fivox::ConstEventSourcePtr loader = source->GetFunctor().getSource();
        const ::fivox::AABBf& bbox = loader->getBoundingBox();
        const vmml::Vector3f& baseSpacing = bbox.getDimension() / info.voxels;
        const int32_t levelFromBottom = node.getMaxRefLevel() -
                                        node.getRefLevel();
        const float spacingFactor = 1 << levelFromBottom;

        Image::SpacingType spacing;
        spacing[0] = baseSpacing.find_max() * spacingFactor;
        spacing[1] = spacing[0];
        spacing[2] = spacing[0];

        const vmml::Vector3f& offset = bbox.getMin() +
                                       node.getRelativePosition() *
                               vmml::Vector3f( bbox.getDimension().find_max( ));
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
        const size_t size = voxels[ 0 ] * voxels[ 1 ] * voxels[ 2 ] *
                            info.compCount * info.getBytesPerVoxel();
        memoryUnit->allocAndSetData( image->GetBufferPointer(), size );
        return memoryUnit;
    }

    SourcePtr source;

private:
    mutable lunchbox::Lock _lock;
};
}

DataSource::DataSource( const ::livre::VolumeDataSourcePluginData& pluginData )
    : _impl( new detail::DataSource( pluginData ))
{
    const lunchbox::URI& uri = pluginData.getURI();
    lunchbox::URI::ConstKVIter i = uri.findQuery( "totalSize" );
    lunchbox::URI::ConstKVIter j = uri.findQuery( "blockSize" );

    const size_t totalSize = (i == uri.queryEnd( )) ? _defaultTotalSize :
                                             lexical_cast< float >( i->second );
    const size_t blockSize = (j == uri.queryEnd( )) ? _defaultBlockSize :
                                             lexical_cast< float >( j->second );

    _volumeInfo.voxels = vmml::Vector3ui( totalSize );
    _volumeInfo.maximumBlockSize = vmml::Vector3ui( blockSize );

    if( !fillRegularVolumeInfo( _volumeInfo ))
       LBTHROW( std::runtime_error( "Cannot setup the regular tree" ));

    ::fivox::ConstEventSourcePtr loader =
          _impl->source->GetFunctor().getSource();
    const ::fivox::AABBf& bbox = loader->getBoundingBox();

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

bool DataSource::handles( const ::livre::VolumeDataSourcePluginData& data )
{
    const std::string fivox = "fivox";
    const std::string& scheme = data.getURI().getScheme();
    return scheme.substr( 0, fivox.size( )) == fivox;
}

}
}
