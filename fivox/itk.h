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

#ifndef FIVOX_ITK_H
#define FIVOX_ITK_H

#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wsign-promo"
#ifdef LB_GCC_4_6_OR_LATER
#  pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#ifdef final
#  undef final
#endif

#define VCL_CAN_STATIC_CONST_INIT_FLOAT 0

#include <itkImageSource.h>
#include <itkNumericTraits.h>
#include <itkImageRegionSplitterDirection.h>

// reset final/override setup
#ifdef LUNCHBOX_COMPILER_H
#  undef LUNCHBOX_COMPILER_H
#endif
#include <lunchbox/compiler.h>

#endif
