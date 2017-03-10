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

#ifndef FIVOX_PROGRESSOBSERVER_H
#define FIVOX_PROGRESSOBSERVER_H

#include <itkCommand.h>
#include <memory>

namespace fivox
{
/**
 * Display the progress of an ITK filter on std::cout and emits a progress event
 * using ZeroEQ (if available).
 */
class ProgressObserver : public itk::Command
{
public:
    typedef itk::SmartPointer<ProgressObserver> Pointer;

    itkNewMacro(ProgressObserver);

    ProgressObserver();

    /** Resets the progress display for a new iteration. */
    void reset();

    /** Enables cout printing of progress */
    void enablePrint();

private:
    class Impl;
    std::unique_ptr<Impl> _impl;

    void Execute(itk::Object* caller, const itk::EventObject& event) override;

    void Execute(const itk::Object* object,
                 const itk::EventObject& event) override;
};

} // end namespace fivox

#endif
