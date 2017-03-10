
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
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

#ifndef FIVOX_FUNCTORIMAGESOURCE_H
#define FIVOX_FUNCTORIMAGESOURCE_H

#include <fivox/imageSource.h>
#include <fivox/types.h>
#include <lunchbox/monitor.h> // member

namespace fivox
{
/** Image source using an EventFunctor on each pixel to generate the output */
template <typename TImage>
class FunctorImageSource : public ImageSource<TImage>
{
public:
    /** Standard class typedefs. */
    typedef FunctorImageSource Self;
    typedef ImageSource<TImage> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;
    typedef EventFunctorPtr<TImage> FunctorPtr;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

        /** Run-time type information (and related methods). */
        itkTypeMacro(FunctorImageSource, ImageSource)

        /** @return the functor executed for each pixel during update. */
        FunctorPtr getFunctor();

    /** Set a new functor. */
    void setFunctor(FunctorPtr functor);

protected:
    FunctorImageSource();
    virtual ~FunctorImageSource() {}
    FunctorImageSource(const FunctorImageSource&) = delete;
    void operator=(const FunctorImageSource&) = delete;

    const itk::ImageRegionSplitterBase* GetImageRegionSplitter() const override
    {
        return _splitter;
    }

    /** FunctorImageSource is implemented as a multithreaded filter. */
    void ThreadedGenerateData(
        const typename Superclass::ImageRegionType& outputRegionForThread,
        itk::ThreadIdType threadId) override;

    void BeforeThreadedGenerateData() override;

private:
    FunctorPtr _functor;
    lunchbox::Monitor<size_t> _completed;
    itk::ImageRegionSplitterBase::Pointer _splitter;
};

} // end namespace fivox

#ifndef ITK_MANUAL_INSTANTIATION
#include "functorImageSource.hxx"
#endif
#endif
