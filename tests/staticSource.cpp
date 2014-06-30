
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#define BOOST_TEST_MODULE StaticSource

#include "test.h"
#include <fivox/imageSource.h>
#include <itkTimeProbe.h>
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

template< typename T, size_t dim >
inline void _testStaticKernel( const size_t size )
{
    typedef itk::Image< T, dim > Image;
    typedef StaticFunctor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    filter->GetFunctor().value = typename Functor::TPixel( 4.2 );
    filter->Update();

    typename Image::IndexType index;
    index.Fill( 0 );

    const typename Image::PixelType& pixel = output->GetPixel( index );
    BOOST_CHECK_EQUAL( pixel, T( 4.2 ));
}
}

BOOST_AUTO_TEST_CASE(StaticSource)
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
                      << i*i*i / 1024.f / 1024.f / clock.GetTotal();
#endif
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testStaticKernel< float, 3 >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << ',' << std::setw(15)
                      << i*i*i / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
#endif
        }
    }
}
