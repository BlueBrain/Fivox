/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *                          Daniel.Nachbaur@epfl.ch
 */

#include "compartmentLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <BBP/BBP.h>

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
    {
        const bbp::Cell_Target& target_ = _reader.getCellTarget();
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );
        _reader.updateMapping( target_ );

        size_t i = 0;
        for( const uint32_t gid : target_ )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            // The voltage reports typically include the soma, dendrites and
            // first two axon sections. Here we will ignore the axon sections.
            const bbp::Sections& sections = neuron.dendrites();

            output.add( Event( neuron.position( ), 0.f ));
            _sections.push_back(
                SectionInfo( 1, _reader.getOffsets()[i][neuron.soma().id()] ));

            for( const bbp::Section& section : sections )
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

            for( uint64_t offset = info.offset; offset < end; ++offset, ++index)
            {
                _output.update( index,
                             ( *voltages )[ offset ] - brion::MINIMUM_VOLTAGE );
            }
        }
        LBINFO << "Updated " << index << " events at " << time << "ms"
               << std::endl;
        return true;
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
};

CompartmentLoader::CompartmentLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new CompartmentLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_reader.getTimestep( ));
}

CompartmentLoader::~CompartmentLoader()
{}

Vector2f CompartmentLoader::_getTimeRange() const
{
    return Vector2f( _impl->_reader.getStartTime(),
                     _impl->_reader.getEndTime( ));
}

bool CompartmentLoader::_load( const float time )
{
    return _impl->load( time );
}

}
