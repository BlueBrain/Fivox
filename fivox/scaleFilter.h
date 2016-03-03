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

#include <fivox/types.h>

#include <itkImageFileWriter.h>
#include <itkIntensityWindowingImageFilter.h>

namespace fivox
{
typedef FloatVolume::Pointer VolumePtr;

template< typename T > class ScaleFilter
{
    typedef itk::IntensityWindowingImageFilter
            < FloatVolume, itk::Image< T, 3 >> IntensityWindowingImageFilter;

public:
    /**
     * Default constructor (used in VolumeWriter< float >)
     */
    ScaleFilter()
        : _scaler( IntensityWindowingImageFilter::New( ))
    {}

    /**
     * ScaleFilter constructor that takes as parameters the volume to be scaled
     * and the input data range
     *
     * @param input Pointer to a floating point volume
     * @param dataWindow Vector2f containing the lower and upper limits for the
     * input data range
     */
    ScaleFilter( VolumePtr input, const Vector2f& dataRange )
    {
        _scaler = IntensityWindowingImageFilter::New();
        _scaler->SetInput( input );

        _scaler->SetWindowMinimum( dataRange[0] );
        _scaler->SetWindowMaximum( dataRange[1] );
        _scaler->SetOutputMinimum( std::numeric_limits< T >::min( ));
        _scaler->SetOutputMaximum( std::numeric_limits< T >::max( ));
    }

    typename IntensityWindowingImageFilter::Pointer operator->()
    {
        return _scaler;
    }

private:
    typename IntensityWindowingImageFilter::Pointer _scaler;
};

}
#endif
