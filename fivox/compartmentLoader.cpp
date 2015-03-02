/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 */

#include "compartmentLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#include <boost/foreach.hpp>

namespace fivox
{
namespace
{
struct SectionInfo
{
  SectionInfo( const size_t nComp, const uint64_t off )
      : numCompartments( nComp )
      , offset( off )
  {}

  size_t numCompartments;
  uint64_t offset;
};

typedef std::vector< SectionInfo > SectionInfos;
}

namespace detail
{
class CompartmentLoader
{
public:
  CompartmentLoader( fivox::EventSource& output, const std::string& blueconfig,
                     const std::string& target, const float time )
      : _output( output )
      , _experiment( blueconfig )
      , _reader( *_experiment.reports().begin(),
                 _experiment.cell_target( target ))
  {
    const bbp::Cell_Target& target_ = _experiment.cell_target( target );
    bbp::Microcircuit& microcircuit = _experiment.microcircuit();
    microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );
    _reader.updateMapping( target_ );

    bbp::CompartmentReportFrame frame;
    if( !_reader.loadFrame( time, frame ))
      throw std::runtime_error( "Can't load compartment report" );

    size_t i=0;
    BOOST_FOREACH( const uint32_t gid, target_ )
    {
      const bbp::Neuron& neuron = microcircuit.neuron( gid );
      const bbp::Sections& sections = neuron.dendrites();
      size_t j = 0;
      BOOST_FOREACH( const bbp::Section& section, sections )
      {
        const size_t nCompartments = _reader.getCompartmentCounts()[i][j];
        const float compartmentLength = 1.f / float( nCompartments );
        const float compartmentMiddle = compartmentLength * .5f;

        for( size_t k = 0; k < nCompartments; ++k )
        {
            const bbp::Cross_Section& midPoint = section.cross_section(
                compartmentMiddle + k * compartmentLength );

            output.add( Event( midPoint.center(), 0.f ));
        }

        _sections.push_back( SectionInfo( nCompartments,
                                          _reader.getOffsets()[i][j] ));
        ++j;
      }
      ++i;
    }

    LBCHECK( loadFrame( time ));
  }

  bool loadFrame( const float time )
  {
    bbp::CompartmentReportFrame frame;
    if( !_reader.loadFrame( time, frame ))
    {
      std::cerr << "Could not load frame at " << time << "ms" << std::endl;
      return false;
    }

    const bbp::floatsPtr& voltages = frame.getData< bbp::floatsPtr >();
    size_t index = 0;
    for( size_t i = 0; i < _sections.size(); ++i )
    {
      const SectionInfo& info = _sections[i];
      const uint64_t end = info.numCompartments + info.offset;

      for( uint64_t offset = info.offset; offset < end; ++offset )
          _output.update( index++, (*voltages)[ offset ] -
                                   brion::RESTING_VOLTAGE );
    }
    LBINFO << "Updated " << index << " events at " << time << std::endl;
    return true;
  }

private:
  fivox::EventSource& _output;
  bbp::Experiment _experiment;
  bbp::CompartmentReportReader _reader;
  SectionInfos _sections;
};
}

CompartmentLoader::CompartmentLoader( const std::string& blueconfig,
                                      const std::string& target,
                                      const float time )
    : _impl( new detail::CompartmentLoader( *this, blueconfig, target, time ))
{}

CompartmentLoader::~CompartmentLoader()
{
  delete _impl;
}

bool CompartmentLoader::loadFrame( const float time )
{
  return _impl->loadFrame( time );
}

}
