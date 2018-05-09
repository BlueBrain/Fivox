
/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
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

#ifndef FIVOX_PROJECTIONIMAGEFILTER_H
#define FIVOX_PROJECTIONIMAGEFILTER_H

#include <itkConceptChecking.h>
#include <itkProjectionImageFilter.h>

#include <fivox/types.h>

/**
 * Filter that uses a Beer-Lambert approach to accumulate and project the
 * values of the input image (e.g. 3D volume) as an output image
 * (e.g. 2D image). Based on itkStandardDeviationProjectionImageFilter.h
 */
namespace Functor
{
template <typename TInputPixel, typename TAccumulate>
class Accumulator
{
public:
    typedef typename itk::NumericTraits<TInputPixel>::RealType RealType;

    explicit Accumulator(itk::SizeValueType) {}
    ~Accumulator() {}
    inline void Initialize()
    {
        _sum = itk::NumericTraits<TAccumulate>::ZeroValue();
    }

    inline void operator()(const TInputPixel& input) { _sum += input; }
    inline RealType GetValue() { return _sum; }
private:
    TAccumulate _sum;
};
} // end namespace Function

template <typename TInputImage, typename TOutputImage,
          typename TAccumulate = typename itk::NumericTraits<
              typename TOutputImage::PixelType>::AccumulateType>
class ProjectionImageFilter
    : public itk::ProjectionImageFilter<
          TInputImage, TOutputImage,
          Functor::Accumulator<typename TInputImage::PixelType, TAccumulate>>
{
public:
    typedef ProjectionImageFilter Self;

    typedef itk::ProjectionImageFilter<
        TInputImage, TOutputImage,
        Functor::Accumulator<typename TInputImage::PixelType, TAccumulate>>
        Superclass;

    typedef TInputImage InputImageType;
    typedef typename InputImageType::PixelType InputPixelType;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    typedef typename Superclass::AccumulatorType AccumulatorType;

    /** Runtime information support. */
    itkTypeMacro(ProjectionImageFilter, ProjectionImageFilter);

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

protected:
    ProjectionImageFilter() {}
    virtual ~ProjectionImageFilter() {}
    virtual AccumulatorType NewAccumulator(itk::SizeValueType size) const
        ITK_OVERRIDE
    {
        AccumulatorType accumulator(size);

        return accumulator;
    }

private:
    ProjectionImageFilter(const Self&) = delete;
    void operator=(const Self&) = delete;
};
#endif
