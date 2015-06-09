// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_SOMALOADER_H
#define FIVOX_SOMALOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
namespace detail { class SomaLoader; }

class SomaLoader : public EventSource
{
public:
  SomaLoader( const std::string& blueconfig, const std::string& target,
              const std::string& report, const float time = 0.f );
  virtual ~SomaLoader();

  bool loadFrame( const float time );

private:
  detail::SomaLoader* const _impl;
};
} // end namespace fivox

#endif
