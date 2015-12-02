/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Daniel.Nachbaur@epfl.ch
 */

#ifndef FIVOX_PROGRESSOBSERVER_H
#define FIVOX_PROGRESSOBSERVER_H

#include <boost/progress.hpp>
#include <itkCommand.h>

namespace fivox
{

/** Displays the progress of an ITK filter on std::cout. */
class ProgressObserver : public itk::Command
{
public:
    itkNewMacro( ProgressObserver );

    ProgressObserver();

    /** Resets the progress display for a new iteration. */
    void reset();

private:
    boost::progress_display _progressBar;
    size_t _previousProgress;

    void Execute( itk::Object* caller, const itk::EventObject& event ) override;

    void Execute( const itk::Object* object,
                  const itk::EventObject& event ) override;
};

} // end namespace fivox

#endif
