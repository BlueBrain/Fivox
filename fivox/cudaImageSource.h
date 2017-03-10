
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

#ifndef FIVOX_CUDAIMAGESOURCE_H
#define FIVOX_CUDAIMAGESOURCE_H

#include <fivox/imageSource.h>
#include <fivox/types.h>

namespace fivox
{
/**
 * Image source using a CUDA compute kernel on each pixel to generate the output
 */
template <typename TImage>
class CudaImageSource : public ImageSource<TImage>
{
public:
    /** Standard class typedefs. */
    typedef CudaImageSource Self;
    typedef ImageSource<TImage> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

        /** Run-time type information (and related methods). */
        itkTypeMacro(CudaImageSource, ImageSource)

            protected : CudaImageSource();
    virtual ~CudaImageSource() {}
    CudaImageSource(const CudaImageSource&) = delete;
    void operator=(const CudaImageSource&) = delete;

    void GenerateData() override;
};

} // end namespace fivox

#ifndef ITK_MANUAL_INSTANTIATION
#include "cudaImageSource.hxx"
#endif
#endif
