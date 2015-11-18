/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *                          Daniel.Nachbaur@epfl.ch
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

class CompartmentLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _experiment( params.getConfig( ))
        , _reader( *_experiment.reports().find( params.getReport( )),
                   _experiment.cell_target(
                       params.getTarget( _experiment.circuit_target( ))))
        , _currentTime( -1.f )
        , _magnitude( params.getMagnitude( ))
    {
        const bbp::Cell_Target& target_ = _reader.getCellTarget();
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );
        _reader.updateMapping( target_ );

        size_t i = 0;
        BOOST_FOREACH( const uint32_t gid, target_ )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            // The voltage reports typically include the soma, dendrites and
            // first two axon sections. Here we will ignore the axon sections.
            const bbp::Sections& sections = neuron.dendrites();

            output.add( Event( neuron.position( ), 0.f ));
            _sections.push_back(
                SectionInfo( 1, _reader.getOffsets()[i][neuron.soma().id()] ));

            BOOST_FOREACH( const bbp::Section& section, sections )
            {
                const uint32_t id = section.id();
                const size_t nCompartments =
                    _reader.getCompartmentCounts()[i][id];
                assert( nCompartments );
                if( !nCompartments )
                    return;
                const float length = 1.f / float( nCompartments );

                for( float k = length * .5f; k < 1.0; k += length )
                    output.add(
                        Event( section.cross_section( k ).center(), 0.f ));

                _sections.push_back( SectionInfo( nCompartments,
                                                  _reader.getOffsets()[i][id] ));
            }
            ++i;
        }
    }

    bool load( const float time )
    {
        if( time == _currentTime )
            return true;
        _currentTime = time;

        bbp::CompartmentReportFrame frame;
        if( !_reader.loadFrame( time, frame ))
        {
            LBERROR << "Could not load frame at " << time << "ms" << std::endl;
            return false;
        }

        const bbp::floatsPtr& voltages = frame.getData< bbp::floatsPtr >();
        size_t index = 0;
        for( size_t i = 0; i < _sections.size(); ++i )
        {
            const SectionInfo& info = _sections[i];
            const uint64_t end = info.numCompartments + info.offset;

            for( uint64_t offset = info.offset; offset < end; ++offset, ++index )
            {
                _output.update( index, _magnitude * (( *voltages )[ offset ] -
                                                     brion::MINIMUM_VOLTAGE ));
            }
        }
        LBINFO << "Updated " << index << " events at " << time << "ms"
               << std::endl;
        return true;
    }

    bool load( const uint32_t frame )
    {
        if( !_output.isInFrameRange( frame ))
            return false;

        const float time  = _reader.getStartTime() + _output.getDt() * frame;
        return load( time );
    }

    Vector2ui getFrameRange()
    {
        const float reportTime = _reader.getEndTime() - _reader.getStartTime();
        const uint32_t numFrames = std::ceil( reportTime / _output.getDt( ));
        return Vector2ui( 0, numFrames );
    }

    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;
    SectionInfos _sections;

    float _currentTime;
    const float _magnitude;
};

CompartmentLoader::CompartmentLoader( const URIHandler& params )
    : _impl( new CompartmentLoader::Impl( *this, params ))
{
    float dt = params.getDt();
    if( dt < 0.f )
         dt = _impl->_reader.getTimestep();
    setDt( dt );
}

CompartmentLoader::~CompartmentLoader()
{}

bool CompartmentLoader::load( const float time )
{
    return _impl->load( time );
}

bool CompartmentLoader::load( const uint32_t frame )
{
    return _impl->load( frame );
}

Vector2ui CompartmentLoader::getFrameRange()
{
    return _impl->getFrameRange();
}

}
