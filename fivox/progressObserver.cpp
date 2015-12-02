/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Daniel.Nachbaur@epfl.ch
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

} // end namespace fivox
