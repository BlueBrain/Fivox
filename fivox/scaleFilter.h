/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
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

#ifndef FIVOX_SCALEFILTER_H
#define FIVOX_SCALEFILTER_H

#include <itkIntensityWindowingImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>

namespace fivox
{

template< typename TImage > class ScaleFilter
{
    typedef fivox::FloatVolume::Pointer VolumePtr;
    typedef itk::IntensityWindowingImageFilter
        < fivox::FloatVolume, TImage > IntensityWindowingImageFilter;

    typedef itk::RescaleIntensityImageFilter< fivox::FloatVolume,
                                              TImage > RescaleFilter;
    typedef typename TImage::PixelType T;

public:
    /**
     * Default constructor (used in VolumeWriter< float >)
     */
    ScaleFilter()
    {}

    /**
     * ScaleFilter constructor that takes as parameters the volume to be scaled
     * and the input data range
     *
     * @param input Pointer to a floating point volume
     * @param dataRange Vector2f containing the lower and upper limits for the
     *                  input data range
     */
    ScaleFilter( VolumePtr input, const fivox::Vector2f& dataRange )
    {
        if( dataRange == fivox::FULLDATARANGE )
        {
            LBINFO << "Scale volume into ["
                   << size_t(std::numeric_limits< T >::min( )) << ", "
                   << size_t(std::numeric_limits< T >::max( ))
                   << "] from data range" << std::endl;
            _rescale = RescaleFilter::New();
            _rescale->SetInput( input );
            return;
        }

        LBINFO << "Scale volume into ["
               << size_t(std::numeric_limits< T >::min( )) << ", "
               << size_t(std::numeric_limits< T >::max( ))
               << "] from values in [" << dataRange[0] << ", " << dataRange[1]
               << "]" << std::endl;

        _scaler = IntensityWindowingImageFilter::New();
        _scaler->SetInput( input );

        _scaler->SetWindowMinimum( dataRange[0] );
        _scaler->SetWindowMaximum( dataRange[1] );
        _scaler->SetOutputMinimum( std::numeric_limits< T >::min( ));
        _scaler->SetOutputMaximum( std::numeric_limits< T >::max( ));
    }

    typename TImage::Pointer GetOutput()
    {
        return _scaler ? _scaler->GetOutput() : _rescale->GetOutput();
    }

    void Update()
    {
        if( _scaler )
            _scaler->Update();
        else
            _rescale->Update();
    }

private:
    typename IntensityWindowingImageFilter::Pointer _scaler;
    typename RescaleFilter::Pointer _rescale;
};

} // namespace fivox

#endif
