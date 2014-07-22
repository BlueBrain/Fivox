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

/** Loads compartment report data to be sampled by an EventFunctor. */
class CompartmentLoader : public EventSource
{
public:
  /**
   * Construct a new compartment event source.
   *
   * @param blueconfig The Blueconfig file for the simulation
   * @param target The target to load
   * @param time The initial frame (time step)
   * @throw H5::exception or std::exception on error
   */
  CompartmentLoader( const std::string& blueconfig,
                     const std::string& target, const float time = 0.f );

  virtual ~CompartmentLoader(); //!< Destruct this compartment event source

  bool loadFrame( const float time ); //!< Load a new time step.

private:
  detail::CompartmentLoader* const _impl;
};
}

#endif
