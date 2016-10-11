
/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Eyescale Software GmbH nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef FIVOX_VOLUMEWRITER_H
#define FIVOX_VOLUMEWRITER_H

#include <fivox/scaleFilter.h>

#include <itkImageFileWriter.h>

typedef fivox::FloatVolume::Pointer VolumePtr;

namespace
{
/**
 * Interface for an itk::ImageFileWriter to write a volume to disk, scaling to
 * a specified data range when using an integer type to specialize the template.
 * Can also write a VSD projection from the same volume.
 */
template< typename T > class VolumeWriter
{
    typedef itk::ImageFileWriter< itk::Image< T, 3 >> Writer;

public:
    /**
     * @param input pointer to the input volume
     * @param dataRange range of the data to be used as reference to scale
     */
    VolumeWriter( VolumePtr input, const vmml::Vector2f& dataRange )
        : _input( input )
        , _scaler( _input, dataRange )
        , _writer( Writer::New( ))
    {
        _writer->SetInput( _scaler.GetOutput( ));
    }

    typename Writer::Pointer operator->() { return _writer; }

private:
    VolumePtr _input;
    fivox::ScaleFilter< itk::Image< T, 3 >> _scaler;
    typename Writer::Pointer _writer;
};

template<> VolumeWriter< float >::VolumeWriter( VolumePtr input,
                                                const vmml::Vector2f& )
    : _input( input )
    , _writer( Writer::New( ))
{
    _writer->SetInput( _input );
}
}
#endif
