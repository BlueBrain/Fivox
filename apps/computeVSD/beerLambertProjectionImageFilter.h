
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

    explicit BeerLambertAccumulator(itk::SizeValueType) {}
    ~BeerLambertAccumulator() {}
    inline void Initialize()
    {
        _sum = itk::NumericTraits<TAccumulate>::ZeroValue();
        _accumulatedValues = 0;
    }

    inline void operator()(const TInputPixel& input)
    {
        // assume that values are being computed with increaing y coordinates
        // in image space, which is bottom to top in circuit space.
        const double height = _pixelSize * _accumulatedValues + _yOrigin;
        const double depth = _circuitHeight - height;
        _sum += input * std::exp(-_sigma * depth);
        ++_accumulatedValues;
    }

    inline RealType GetValue() { return _sum; }
    double _pixelSize;
    double _sigma;
    double _yOrigin;
    double _circuitHeight;

private:
    TAccumulate _sum;
    size_t _accumulatedValues;
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

    /** Set the y position of the volume origin in circuit coordinates.
     * Defaults to 0. */
    itkSetMacro(yOrigin, double);

    /** Set the position of the circuit top in circuit coordinates. Used
        to compute depth measured from top to bottom.
        Defaults to 2081.75. */
    itkSetMacro(CircuitHeight, double);

    /** Get the absorption + scattering coefficient (units per micrometer).
     * Defaults to 1.0. */
    itkGetConstMacro(Sigma, double);

protected:
    BeerLambertProjectionImageFilter()
        : m_PixelSize(1.0)
        , m_Sigma(1.0)
        , m_yOrigin(0.0)
        , m_CircuitHeight(2081.75)
    {
    }

    virtual ~BeerLambertProjectionImageFilter() {}
    virtual AccumulatorType NewAccumulator(itk::SizeValueType size) const
        ITK_OVERRIDE
    {
        AccumulatorType accumulator(size);

        accumulator._pixelSize = m_PixelSize;
        accumulator._sigma = m_Sigma;
        accumulator._yOrigin = m_yOrigin;
        accumulator._circuitHeight = m_CircuitHeight;
        return accumulator;
    }

private:
    BeerLambertProjectionImageFilter(const Self&) = delete;
    void operator=(const Self&) = delete;

    /** Micrometers per input pixel */
    double m_PixelSize;

    /** Absorption + scattering coefficient (units per micrometer) */
    double m_Sigma;

    double m_yOrigin;
    double m_CircuitHeight;
};
#endif
