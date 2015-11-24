/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
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
