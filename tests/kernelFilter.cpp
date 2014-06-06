
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#define BOOST_TEST_MODULE KernelFilter
#ifndef FIVOX_USE_BBPSDK
#  ifdef FIVOX_USE_BBPTESTDATA
#    undef FIVOX_USE_BBPTESTDATA
#  endif
#endif

#include <fivox/compartmentFunctor.h>
#include <fivox/imageSource.h>
#ifdef FIVOX_USE_BBPTESTDATA
#  include <BBP/TestDatasets.h>
#endif
#include <itkTimeProbe.h>
#include <boost/make_shared.hpp>
#include <boost/test/unit_test.hpp>
#include <iomanip>

#ifdef NDEBUG
static const size_t maxSize = 1024;
#else
static const size_t maxSize = 128;
#endif

namespace
{
template< typename TImage > class StaticFunctor
{
public:
    typedef typename TImage::PixelType TPixel;
    typedef typename TImage::PointType TPoint;

    StaticFunctor() {}
    ~StaticFunctor() {}

    bool operator!=(const StaticFunctor &) const { return false; }
    bool operator==(const StaticFunctor & other) const
        { return !( *this != other ); }

    inline TPixel operator()( const TPoint& /*point*/ ) const
    {
        return value;
    }

    TPixel value;
};

template< typename TImage >
inline void _setSize( typename TImage::Pointer image, const size_t size )
{
    typename TImage::SizeType vSize;
    vSize.Fill( size );

    typename TImage::RegionType region;
    region.SetSize( vSize );

    image->SetRegions( region );
}

template< typename T, size_t dim >
inline void _testStaticKernel( const size_t size )
{
    typedef itk::Image< T, dim > Image;
    typedef StaticFunctor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    filter->GetFunctor().value = 4.2;
    filter->Update();

    typename Image::IndexType index;
    index.Fill( 0 );

    const typename Image::PixelType& pixel = output->GetPixel( index );
    BOOST_CHECK_EQUAL( pixel, T( 4.2 ));
}

}

BOOST_AUTO_TEST_CASE(KernelFilter)
{
#ifdef NDEBUG
    std::cout.setf( std::ios::right, std::ios::adjustfield );
    std::cout.precision( 5 );
    std::cout << "Static fill, byte MVox/sec, float MVox/sec" << std::endl;
#endif

    for( size_t i = 1; i <= maxSize; i = i << 1 )
    {
        {
            itk::TimeProbe clock;
            clock.Start();
            _testStaticKernel< unsigned char, 3 >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << std::setw( 11 ) << i << ',' << std::setw(14)
                      << std::pow( i, 3 ) / 1024.f / 1024.f / clock.GetTotal();
#endif
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testStaticKernel< float, 3 >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << ',' << std::setw(15)
                      << std::pow( i, 3 ) / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
#endif
        }
    }
}

#ifdef FIVOX_USE_BBPTESTDATA
namespace
{
template< typename T >
inline void _testSDKKernel( const size_t size )
{
    typedef itk::Image< T, 3 > Image;
    typedef fivox::CompartmentFunctor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    filter->GetFunctor().setLoader(
        boost::make_shared< fivox::CompartmentLoader >(
            bbp::test::getBlueconfig(), "L5CSPC" ));
    filter->Update();
}

}

BOOST_AUTO_TEST_CASE(SDKFilter)
{
#ifdef NDEBUG
    std::cout.setf( std::ios::right, std::ios::adjustfield );
    std::cout.precision( 5 );
    std::cout << " SDK Kernel, byte MVox/sec, float MVox/sec" << std::endl;
#endif

    for( size_t i = 1; i <= maxSize; i = i << 1 )
    {
        {
            itk::TimeProbe clock;
            clock.Start();
            _testSDKKernel< float >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << std::setw( 11 ) << i << ',' << std::setw(14)
                      << std::pow( i, 3 ) / 1024.f / 1024.f / clock.GetTotal();
#endif
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testSDKKernel< float >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << ',' << std::setw(15)
                      << std::pow( i, 3 ) / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
#endif
        }
    }
}
#endif
