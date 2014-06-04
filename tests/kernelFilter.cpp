
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#define BOOST_TEST_MODULE KernelFilter

#include <fivox/imageSource.h>
#include <itkTimeProbe.h>
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
    typedef typename itk::NumericTraits< TPixel >::AccumulateType TAccumulator;

    StaticFunctor() {}
    ~StaticFunctor() {}

    bool operator!=(const StaticFunctor &) const { return false; }
    bool operator==(const StaticFunctor & other) const
        { return !( *this != other ); }

    inline TPixel operator()( const TPoint& /*point*/ ) const
    {
        //const TAccumulator sum = value;
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
    typedef T Pixel;
    typedef itk::Image< Pixel, dim > Image;
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
    std::cout.setf( std::ios::right, std::ios::adjustfield );
    std::cout.precision( 5 );
    std::cout << "Volume size, byte GVox/sec, float GVox/sec" << std::endl;

    for( size_t i = 1; i <= maxSize; i = i << 1 )
    {
        {
            itk::TimeProbe clock;
            clock.Start();
            _testStaticKernel< unsigned char, 3 >( i );
            clock.Stop();
            std::cout << std::setw( 11 ) << i << ',' << std::setw(14)
                      << std::pow( i, 3 ) / 1024.f / 1024.f / clock.GetTotal();
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testStaticKernel< float, 3 >( i );
            clock.Stop();
            std::cout << ',' << std::setw(15)
                      << std::pow( i, 3 ) / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
        }
    }
}
