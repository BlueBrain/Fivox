
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Eyescale Software GmbH nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define BOOST_TEST_MODULE EventFunctor

#include "test.h"
#include <fivox/eventFunctor.h>
#include <fivox/eventSource.h>
#include <fivox/functorImageSource.h>
#include <iomanip>
#include <itkTimeProbe.h>

#ifdef NDEBUG
static const size_t maxSize = 512;
#else
static const size_t maxSize = 128;
#endif

namespace
{
template <class TImage>
class MeaningFunctor : public fivox::EventFunctor<TImage>
{
    typedef fivox::EventFunctor<TImage> Super;

public:
    MeaningFunctor()
        : Super()
    {
    }
    virtual ~MeaningFunctor() {}
    typename Super::TPixel operator()(const typename Super::TPoint&,
                                      const typename Super::TSpacing&) const
    {
        return 42.f;
    }
};

template <typename T, size_t dim>
inline void _testEventFunctor(const size_t size)
{
    typedef itk::Image<T, dim> Image;
    typedef MeaningFunctor<Image> Functor;
    typedef fivox::FunctorImageSource<Image> Filter;

    typename Filter::Pointer filter = Filter::New();
    typename Image::Pointer output = filter->GetOutput();
    _setSize<Image>(output, size);

    filter->setFunctor(std::make_shared<Functor>());
    filter->Update();

    typename Image::IndexType index;
    index.Fill(size / 2);

    const typename Image::PixelType& pixel = output->GetPixel(index);
    BOOST_CHECK_EQUAL(pixel, T(42.f));
}
}

BOOST_AUTO_TEST_CASE(EventFunctor)
{
#ifdef NDEBUG
    std::cout.setf(std::ios::right, std::ios::adjustfield);
    std::cout.precision(5);
    std::cout << "Static fill, byte MVox/sec, float MVox/sec" << std::endl;
#endif

    for (size_t i = 1; i <= maxSize; i = i << 1)
    {
        {
            itk::TimeProbe clock;
            clock.Start();
            _testEventFunctor<unsigned char, 3>(i);
            clock.Stop();
#ifdef NDEBUG
            std::cout << std::setw(11) << i << ',' << std::setw(14)
                      << i * i * i / 1024.f / 1024.f / clock.GetTotal();
#endif
        }
        {
            itk::TimeProbe clock;
            clock.Start();
            _testEventFunctor<float, 3>(i);
            clock.Stop();
#ifdef NDEBUG
            std::cout << ',' << std::setw(15)
                      << i * i * i / 1024.f / 1024.f / clock.GetTotal()
                      << std::endl;
#endif
        }
    }
}
