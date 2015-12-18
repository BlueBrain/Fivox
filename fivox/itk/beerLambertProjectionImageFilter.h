/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef FIVOX_BEERLAMBERTPROJECTIONIMAGEFILTER_H
#define FIVOX_BEERLAMBERTPROJECTIONIMAGEFILTER_H

#include <itkProjectionImageFilter.h>
#include <itkConceptChecking.h>

#include <fivox/types.h>

namespace fivox
{
/**
 * Filter that uses a Beer-Lambert approach to accumulate and project the
 * values of the input image (e.g. 3D volume) as an output image
 * (e.g. 2D image). Based on itkStandardDeviationProjectionImageFilter.h
 */
namespace Functor
{
template< typename TInputPixel, typename TAccumulate >
class BeerLambertAccumulator
{

public:
    typedef typename itk::NumericTraits< TInputPixel >::RealType RealType;

    BeerLambertAccumulator( itk::SizeValueType size )
    {
        m_Size = size;
        m_Values.reserve( size );
    }

    ~BeerLambertAccumulator(){}

    inline void Initialize()
    {
        m_Sum = itk::NumericTraits< TAccumulate >::ZeroValue();
        m_Values.clear();
    }

    inline void operator()( const TInputPixel& input )
    {
        // assume that values are being computed in order (top-down)
        const double depth =
                m_PixelSize * ( m_Size - m_Values.size( )) / (double)m_Size;

        m_Sum += input * std::exp( -m_Sigma * depth );
        m_Values.push_back( input );
    }

    inline RealType GetValue()
    {
        return m_Sum;
    }

    double m_PixelSize;
    double m_Sigma;

private:
    TAccumulate m_Sum;
    itk::SizeValueType m_Size;
    std::vector< TInputPixel > m_Values;
};
} // end namespace Function

template< typename TInputImage, typename TOutputImage,
          typename TAccumulate =
          typename itk::NumericTraits<
              typename TOutputImage::PixelType >::AccumulateType >
class BeerLambertProjectionImageFilter :
        public itk::ProjectionImageFilter< TInputImage, TOutputImage,
            Functor::BeerLambertAccumulator< typename TInputImage::PixelType,
                                             TAccumulate > >
{
public:
    typedef BeerLambertProjectionImageFilter Self;

    typedef itk::ProjectionImageFilter< TInputImage, TOutputImage,
        Functor::BeerLambertAccumulator< typename TInputImage::PixelType,
                                         TAccumulate > > Superclass;

    typedef TInputImage InputImageType;
    typedef typename InputImageType::PixelType InputPixelType;

    typedef itk::SmartPointer< Self > Pointer;
    typedef itk::SmartPointer< const Self > ConstPointer;

    typedef typename Superclass::AccumulatorType AccumulatorType;

    /** Runtime information support. */
    itkTypeMacro( BeerLambertProjectionImageFilter, ProjectionImageFilter );

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Set the size (in micrometers) of each input pixel. Defaults to 1.0. */
    itkSetMacro( PixelSize, double );

    /** Get the size (in micrometers) of each input pixel. Defaults to 1.0. */
    itkGetConstMacro( PixelSize, double );

    /** Set the absorption + scattering coefficient (units per micrometer).
     * Defaults to 1.0. */
    itkSetMacro( Sigma, double );

    /** Get the absorption + scattering coefficient (units per micrometer).
     * Defaults to 1.0. */
    itkGetConstMacro( Sigma, double );

protected:
    BeerLambertProjectionImageFilter()
        : m_PixelSize( 1.0 )
        , m_Sigma( 1.0 )
    {}

    virtual ~BeerLambertProjectionImageFilter() {}

    virtual AccumulatorType NewAccumulator( itk::SizeValueType size ) const
        ITK_OVERRIDE
    {
        AccumulatorType accumulator( size );

        accumulator.m_PixelSize = m_PixelSize;
        accumulator.m_Sigma = m_Sigma;
        return accumulator;
    }

private:
    //purposely not implemented
    BeerLambertProjectionImageFilter( const Self & );

    void operator=( const Self & ); //purposely not implemented

    /** Micrometers per input pixel */
    double m_PixelSize;

    /** Absorption + scattering coefficient (units per micrometer) */
    double m_Sigma;
};
}

#endif
