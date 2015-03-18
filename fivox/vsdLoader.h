/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */
#ifndef FIVOX_VSDLOADER_H
#define FIVOX_VSDLOADER_H

#include <fivox/eventSource.h> // base class

namespace fivox
{
namespace detail { class VSDLoader; }

/**
 * Loads report data for voltage sensitive dye sampling.
 *
 * Expects an 'area' full compartment report with a single time step containing
 * the surface areas of each compartment, and a 'voltages' soma report
 * containing the soma voltages for the given time step.
 */
class VSDLoader : public EventSource
{
public:
  /**
   * Construct a new vsd event source.
   *
   * @param blueconfig The Blueconfig file for the simulation
   * @param target The target to load
   * @param time The initial frame (time step)
   * @throw H5::exception or std::exception on error
   */
  VSDLoader( const std::string& blueconfig, const std::string& target,
             const float time = 0.f );

  virtual ~VSDLoader(); //!< Destruct this vsd event source

  bool loadFrame( const float time ); //!< Load a new time step.

private:
  detail::VSDLoader* const _impl;
};
}

#endif
