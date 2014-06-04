
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

//#include <fivox/binarySphereFunctor.h>
#include <fivox/imageSource.h>

#define BOOST_TEST_MODULE KernelFilter
#include <boost/test/unit_test.hpp>


namespace
{
template< typename TImage > class Functor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;

    Functor() {}
    ~Functor() {}

    bool operator!=(const Functor &) const { return false; }
    bool operator==(const Functor & other) const
        { return !( *this != other ); }

    inline TPixel operator()( const TPoint& /*point*/ ) const
    {
        //const TAccumulator sum = value;
        return value;
    }

    float value;
};
}

BOOST_AUTO_TEST_CASE(KernelFilter)
{
    typedef unsigned char Pixel;
    typedef itk::Image< Pixel, 3 > Image;
    //typedef fivox::BinaryKernel< Image::SizeType > Kernel;
    //typedef fivox::KernelFilter< Image, Kernel > Filter;
    typedef Functor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;
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
    output->SetRegions( region );

    filter->GetFunctor().value =42.f;
    filter->Update();

    Image::IndexType index;
    index[0] = 1;
    index[1] = 2;
    index[2] = 3;
    const Image::PixelType& pixel = output->GetPixel( index );
    BOOST_CHECK_EQUAL( pixel, 42.f );
}
