// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_COMPARTMENTLOADER_H
#define FIVOX_COMPARTMENTLOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
namespace detail { class CompartmentLoader; }

class CompartmentLoader : public EventSource
{
public:
  CompartmentLoader( const std::string& blueconfig,
                     const std::string& target, const float time = 0.f );
  virtual ~CompartmentLoader();

  bool loadFrame( const float time );

private:
  detail::CompartmentLoader* const _impl;
};
} // end namespace fivox

#endif
