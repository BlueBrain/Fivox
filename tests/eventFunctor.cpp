
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#define BOOST_TEST_MODULE EventFunctor

#include "test.h"
#include <fivox/eventSource.h>
#include <fivox/imageSource.h>
#include <fivox/eventFunctor.h>
#include <itkTimeProbe.h>
#include <iomanip>

#ifdef NDEBUG
static const size_t maxSize = 512;
#else
static const size_t maxSize = 128;
#endif

namespace
{

template< class Image >
typename Image::PixelType meaningOfEverything(
                         const fivox::EventSource&,
                         const typename Image::PointType&)
{

     return 42.f;
}
template< class TImage >
class MeaningFunctor : public fivox::EventFunctor< TImage >
{
    typedef fivox::EventFunctor< TImage > Super;
public:
    MeaningFunctor() {}
    virtual ~MeaningFunctor() {}

    typename Super::TPixel operator()( const typename Super::TPoint&  )
        const final
    {
        return 42.f;
    }
};

template< typename T, size_t dim >
inline void _testEventFunctor( const size_t size )
{
    typedef itk::Image< T, dim > Image;
    typedef MeaningFunctor< Image > Functor;
    typedef fivox::ImageSource< Image, Functor > Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize< Image >( output, size );

    filter->Update();

    typename Image::IndexType index;
    index.Fill( 0 );

    const typename Image::PixelType& pixel = output->GetPixel( index );
    BOOST_CHECK_EQUAL( pixel, T(  42.f ));
}
}

BOOST_AUTO_TEST_CASE(EventFunctor)
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
            _testEventFunctor< unsigned char, 3 >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << std::setw( 11 ) << i << ',' << std::setw(14)
                      << i*i*i / 1024.f / 1024.f / clock.GetTotal();
#endif
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testEventFunctor< float, 3 >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << ',' << std::setw(15)
                      << i*i*i / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
#endif
        }
    }
}
