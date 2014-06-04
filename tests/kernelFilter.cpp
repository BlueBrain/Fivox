
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

//#include <fivox/binarySphereFunctor.h>
#include <fivox/imageSource.h>

#define BOOST_TEST_MODULE KernelFilter
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(KernelFilter)
{
    typedef unsigned char Pixel;
    typedef itk::Image< Pixel, 3 > Image;
    //typedef fivox::BinaryKernel< Image::SizeType > Kernel;
    //typedef fivox::KernelFilter< Image, Kernel > Filter;
    typedef fivox::ImageSource< Image > Filter;
    static const size_t size = 32;

    Image::SizeType vSize;
    vSize.Fill( size );

    Image::RegionType region;
    region.SetSize( vSize );

    // Kernel::SizeType radius;
    // radius.Fill( 5.f );

    // Kernel::SizeType position;
    // position.Fill( 16.f );

    // Kernel kernel;
    // kernel.setRadius( radius );
    // kernel.addEvent( position, .5f );

    // position.Fill( 12.f );
    // kernel.addEvent( position, 2.f );

    Filter::Pointer filter = Filter::New();
    Image::Pointer output = filter->GetOutput();
    output->SetRequestedRegion( region );
    //filter->SetKernel( kernel );

    filter->Update();

    //BOOST_CHECK_EQUAL( ret, 3 );
}
