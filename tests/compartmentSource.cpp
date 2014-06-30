
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#if defined(FIVOX_USE_BBPSDK) && defined(FIVOX_USE_BBPTESTDATA)

#define BOOST_TEST_MODULE SomaSource

#include "sdk.h"
#include <fivox/compartmentLoader.h>
#include <itkTimeProbe.h>
#include <iomanip>

#ifdef NDEBUG
static const size_t maxSize = 512;
#else
static const size_t maxSize = 64;
#endif

BOOST_AUTO_TEST_CASE(CompartmentSource)
{
#ifdef NDEBUG
    std::cout.setf( std::ios::right, std::ios::adjustfield );
    std::cout.precision( 5 );
    std::cout << "Comp Kernel, byte MVox/sec, float MVox/sec" << std::endl;
#endif

    for( size_t i = 1; i <= maxSize; i = i << 1 )
    {
        {
            itk::TimeProbe clock;
            clock.Start();
            _testSDKKernel< unsigned char, fivox::CompartmentLoader >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << std::setw( 11 ) << i << ',' << std::setw(14)
                      << i*i*i / 1024.f / 1024.f / clock.GetTotal();
#endif
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testSDKKernel< float, fivox::CompartmentLoader >( i );
            clock.Stop();
#ifdef NDEBUG
            std::cout << ',' << std::setw(15)
                      << i*i*i / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
#endif
        }
    }
}

#endif
