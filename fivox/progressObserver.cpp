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

#include <boost/progress.hpp>
#include <itkProcessObject.h>

#ifdef USE_ZEROEQ_PROGRESS
#include <lexis/data/Progress.h>
#include <zeroeq/publisher.h>
#endif

namespace fivox
{
// arbitrary resolution for 0..1 range reported by ITK
static const size_t _expectedCount = 1000000;

class ProgressObserver::Impl
{
public:
    Impl()
        : previousProgress(0)
#ifdef USE_ZEROEQ_PROGRESS
        , progressEvent(
              lexis::data::Progress("Sampling volume", _expectedCount))
#endif
    {
    }

    std::unique_ptr<boost::progress_display> progressBar;
    size_t previousProgress;
#ifdef USE_ZEROEQ_PROGRESS
    lexis::data::Progress progressEvent;
    zeroeq::Publisher publisher;
#endif
};

ProgressObserver::ProgressObserver()
    : _impl(new Impl)
{
}

void ProgressObserver::reset()
{
    if (_impl->progressBar)
        _impl->progressBar->restart(_expectedCount);
#ifdef USE_ZEROEQ_PROGRESS
    _impl->progressEvent.restart(_expectedCount);
#endif
    _impl->previousProgress = 0;
}

void ProgressObserver::enablePrint()
{
    if (!_impl->progressBar)
        _impl->progressBar.reset(new boost::progress_display(_expectedCount));
}

void ProgressObserver::Execute(itk::Object* caller,
                               const itk::EventObject& event)
{
    Execute((const itk::Object*)caller, event);
}

void ProgressObserver::Execute(const itk::Object* object,
                               const itk::EventObject& event)
{
    const itk::ProcessObject* filter =
        static_cast<const itk::ProcessObject*>(object);
    if (!itk::ProgressEvent().CheckEvent(&event))
        return;

    const size_t progress = std::floor(_expectedCount * filter->GetProgress());
    if (_impl->progressBar)
        *_impl->progressBar += progress - _impl->previousProgress;
#ifdef USE_ZEROEQ_PROGRESS
    _impl->progressEvent += progress - _impl->previousProgress;
    _impl->publisher.publish(_impl->progressEvent);
#endif
    _impl->previousProgress = progress;
}
}
