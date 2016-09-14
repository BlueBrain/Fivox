
/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Ahmet.Bilgili@epfl.ch
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

#ifndef FIVOX_ATTENUATION_CURVE_H
#define FIVOX_ATTENUATION_CURVE_H

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

namespace fivox
{

/** Attentuation curve access for depth-based interpolation. @sa VSDLoader */
class AttenuationCurve
{
public:
    /** Construct an empty attenuation curve */
    AttenuationCurve() : _thickness( 1.f ) {}

    /**
     * @param dyeCurveFile Dye attenuation file.
     * @param thickness Thickness of the circuit.
     */
    AttenuationCurve( const std::string& dyeCurveFile, const float thickness )
        : _thickness( thickness )
    {
        if( dyeCurveFile.empty())
            return;

        std::ifstream ifs( dyeCurveFile );

        if( !ifs.is_open( ))
            return;

        std::string line;
        while( std::getline( ifs, line ))
            _dyeCurve.push_back( atof( line.c_str( )));

        if( _dyeCurve.empty())
            return;

        const float maxAttn = *std::max_element( _dyeCurve.begin(),
                                                 _dyeCurve.end());
        for( float& dyeAttn : _dyeCurve )
            dyeAttn = dyeAttn / maxAttn;
    }

    /**
     * Get the attenuation for the given Y-coordinate.
     *
     * If no curve file was loaded, returns 1.0f for any attenuation depth so it
     * does not modify the value of event. If yCoord is larger than the
     * thickness it returns the first element of the curve (generally 0.f)
     *
     * @param yCoord Absolute position (Y axis) of the point.
     * @param interpolate If true, interpolate the attenuation between the two
     * closest indices in the attenuation curve.
     * @return the interpolated attenuation value according to depth; 1 if there
     * is no attenuation curve or it is empty.
     */
    float getAttenuation( const float yCoord,
                          const bool interpolate = false ) const
    {
        if( _dyeCurve.empty( ))
            return 1.0f;

        if( yCoord >= _thickness )
            return _dyeCurve[0];

        const float depth = _thickness - yCoord;
        if( _dyeCurve.size() == 1 || depth >= _thickness )
            return _dyeCurve[ _dyeCurve.size() - 1 ];

        const float deltaDepthPerAttn = _thickness / ( _dyeCurve.size() - 1 );
        const size_t index = depth / deltaDepthPerAttn;
        const float diffToIndexRatio = (depth - deltaDepthPerAttn * index)
                                     / deltaDepthPerAttn;

        const float attenuation = _dyeCurve[ index + 1 ];
        return interpolate ? diffToIndexRatio * attenuation +
                            ( 1.0f - diffToIndexRatio ) * _dyeCurve[ index ]
                           : attenuation;
    }

private:
    std::vector<float> _dyeCurve;
    float _thickness;
};

}
#endif
