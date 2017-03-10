
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

#ifndef FIVOX_BEERLAMBERTPROJECTIONIMAGEFILTER_H
#define FIVOX_BEERLAMBERTPROJECTIONIMAGEFILTER_H

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
class BeerLambertAccumulator
{
public:
    typedef typename itk::NumericTraits<TInputPixel>::RealType RealType;

    explicit BeerLambertAccumulator(itk::SizeValueType size)
    {
        m_Size = size;
        m_Values.reserve(size);
    }

    ~BeerLambertAccumulator() {}
    inline void Initialize()
    {
        m_Sum = itk::NumericTraits<TAccumulate>::ZeroValue();
        m_Values.clear();
    }

    inline void operator()(const TInputPixel& input)
    {
        // assume that values are being computed in order (top-down)
        const double depth = m_PixelSize * m_Values.size();

        m_Sum += input * std::exp(-m_Sigma * depth);
        m_Values.push_back(input);
    }

    inline RealType GetValue() { return m_Sum; }
    double m_PixelSize;
    double m_Sigma;

private:
    TAccumulate m_Sum;
    itk::SizeValueType m_Size;
    std::vector<TInputPixel> m_Values;
};
} // end namespace Function

template <typename TInputImage, typename TOutputImage,
          typename TAccumulate = typename itk::NumericTraits<
              typename TOutputImage::PixelType>::AccumulateType>
class BeerLambertProjectionImageFilter
    : public itk::ProjectionImageFilter<
          TInputImage, TOutputImage,
          Functor::BeerLambertAccumulator<typename TInputImage::PixelType,
                                          TAccumulate>>
{
public:
    typedef BeerLambertProjectionImageFilter Self;

    typedef itk::ProjectionImageFilter<
        TInputImage, TOutputImage,
        Functor::BeerLambertAccumulator<typename TInputImage::PixelType,
                                        TAccumulate>>
        Superclass;

    typedef TInputImage InputImageType;
    typedef typename InputImageType::PixelType InputPixelType;

    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    typedef typename Superclass::AccumulatorType AccumulatorType;

    /** Runtime information support. */
    itkTypeMacro(BeerLambertProjectionImageFilter, ProjectionImageFilter);

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Set the size (in micrometers) of each input pixel. Defaults to 1.0. */
    itkSetMacro(PixelSize, double);

    /** Get the size (in micrometers) of each input pixel. Defaults to 1.0. */
    itkGetConstMacro(PixelSize, double);

    /** Set the absorption + scattering coefficient (units per micrometer).
     * Defaults to 1.0. */
    itkSetMacro(Sigma, double);

    /** Get the absorption + scattering coefficient (units per micrometer).
     * Defaults to 1.0. */
    itkGetConstMacro(Sigma, double);

protected:
    BeerLambertProjectionImageFilter()
        : m_PixelSize(1.0)
        , m_Sigma(1.0)
    {
    }

    virtual ~BeerLambertProjectionImageFilter() {}
    virtual AccumulatorType NewAccumulator(itk::SizeValueType size) const
        ITK_OVERRIDE
    {
        AccumulatorType accumulator(size);

        accumulator.m_PixelSize = m_PixelSize;
        accumulator.m_Sigma = m_Sigma;
        return accumulator;
    }

private:
    BeerLambertProjectionImageFilter(const Self&) = delete;
    void operator=(const Self&) = delete;

    /** Micrometers per input pixel */
    double m_PixelSize;

    /** Absorption + scattering coefficient (units per micrometer) */
    double m_Sigma;
};
#endif
