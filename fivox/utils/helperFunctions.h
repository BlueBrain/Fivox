
/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

namespace fivox
{
template< class T > inline T scale( const float value )
    { return value; }
template<> inline unsigned char scale( const float value )
    { return std::min( value * 256.f, 255.f ); }
}

#endif // HELPER_FUNCTIONS_H

