
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include <boost/test/unit_test.hpp>

namespace
{
template< typename TImage >
inline void _setSize( typename TImage::Pointer image, const size_t size )
{
    typename TImage::SizeType vSize;
    vSize.Fill( size );

    typename TImage::RegionType region;
    region.SetSize( vSize );

    image->SetRegions( region );
}
}
