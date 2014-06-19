// -*- c-basic-offset: 2 -*-

/* Copyright (c) 2014, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "compartmentLoader.h"

#include <BBP/BBP.h>
#ifdef final
#  undef final
#endif

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
  CompartmentLoader( EventSource& output, const std::string& blueconfig,
                     const std::string& target, const float time )
      : _output( output )
      , _experiment( blueconfig )
      , _reader( *_experiment.reports().find( "allCompartments" ),
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
        const bbp::Segments& segments = section.segments();
        Vector3f position( 0.f );

        BOOST_FOREACH( const bbp::Segment& segment, segments )
          position += segment.begin().center() + segment.end().center();
        position /= segments.size() * 2;

        output.add( Event( position, 0.f ));
        _sections.push_back( SectionInfo( _reader.getCompartmentCounts()[i][j],
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
    for( size_t i = 0; i < _sections.size(); ++i )
    {
      const SectionInfo& info = _sections[i];
      const uint64_t end = info.numCompartments + info.offset;
      float voltage = 0.f;
      for( uint64_t offset = info.offset; offset < end; ++offset )
        voltage += (*voltages)[ offset ];

      _output.updateValue( i, voltage / float( info.numCompartments ) + 65.f );
    }
    return true;
  }

private:
  EventSource& _output;
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
