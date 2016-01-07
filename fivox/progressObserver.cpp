/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Daniel.Nachbaur@epfl.ch
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

#include "progressObserver.h"

#include <itkProcessObject.h>

namespace fivox
{

// arbitrary resolution for 0..1 range reported by ITK
const size_t expectedCount = 1000000;

ProgressObserver::ProgressObserver()
    : _progressBar ( expectedCount )
    , _previousProgress( 0 )
{}

void ProgressObserver::reset()
{
    _progressBar.restart( expectedCount );
    _previousProgress = 0;
}

void ProgressObserver::Execute( itk::Object* caller,
                                const itk::EventObject& event )
{
    Execute( (const itk::Object *)caller, event );
}

void ProgressObserver::Execute( const itk::Object* object,
                                const itk::EventObject& event )
{
    const itk::ProcessObject* filter =
            static_cast< const itk::ProcessObject* >( object );
    if( !itk::ProgressEvent().CheckEvent( &event ))
        return;

    const size_t progress = std::floor( expectedCount * filter->GetProgress( ));
    _progressBar +=  progress - _previousProgress;
    _previousProgress = progress;
}

}
