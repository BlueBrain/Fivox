
/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Ahmet.Bilgili@epfl.ch
 *                          Stefan.Eilemann@epfl.ch
 */

#ifndef FIVOX_ATTENUATION_CURVE_H
#define FIVOX_ATTENUATION_CURVE_H

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

namespace fivox
{
/** Attentuation curve access for depth-based interpolation. */
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
     * Get the attenuation for the given depth.
     *
     * If no curve file was loaded, returns 1.0f for any attenuation depth so it
     * does not modify the value of event. If depth is larger than the
     * thickness it returns the first element of the curve (generally 0.f)
     *
     * @param depth Absolute depth (Y axis) of the point.
     * @return the interpolated attenuation value according to depth; 1 if there
     * is no attenuation curve or it is empty.
     */
    float getAttenuation( const float depth ) const
    {
        if( _dyeCurve.empty( ))
            return 1.0f;

        if( _dyeCurve.size() == 1 || depth >= _thickness )
            return _dyeCurve[ 0 ];

        const float invertedDepth = _thickness - depth;
        const float deltaDepthPerAttn = _thickness / ( _dyeCurve.size() - 1 );
        const size_t ind = invertedDepth / deltaDepthPerAttn;
        const float diffToIndRatio = (invertedDepth - deltaDepthPerAttn * ind)
                                                        / deltaDepthPerAttn;
        return diffToIndRatio * _dyeCurve[ ind + 1 ] +
               ( 1.0f - diffToIndRatio ) * _dyeCurve[ ind ];
    }

private:
    std::vector<float> _dyeCurve;
    float _thickness;
};

}

#endif
