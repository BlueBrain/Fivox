/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
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
        , _currentFrameId( 0xFFFFFFFFu )
        , _dt( params.getDt( ))
    {
        const bbp::Cell_Target& target_ = _reader.getCellTarget();
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( target_, bbp::NEURONS | bbp::MORPHOLOGIES );
        _reader.updateMapping( target_ );

        if( _dt < 0.f )
            _dt = _reader.getTimestep();

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
    }

    bool load( const float time )
    {
        static const float magnitude = 0.1f; // heuristic

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

            for( uint64_t offset = info.offset; offset < end; ++offset )
                _output.update( index++, magnitude * (( *voltages )[ offset ] -
                                                      brion::RESTING_VOLTAGE ));
        }
        LBINFO << "Updated " << index << " events at " << time << "ms"
               << std::endl;
        return true;
    }

    void load( const uint32_t frame )
    {
        if( frame == _currentFrameId )
            return;

        _currentFrameId = frame;
        const float time  = _reader.getStartTime() + _dt * frame;
        LBCHECK( load( time ));
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;
    SectionInfos _sections;

    uint32_t _currentFrameId;
    float _dt;
};

CompartmentLoader::CompartmentLoader( const URIHandler& params )
    : _impl( new CompartmentLoader::Impl( *this, params ))
{}

CompartmentLoader::~CompartmentLoader()
{}

void CompartmentLoader::load( const float time )
{
    _impl->load( time );
}

void CompartmentLoader::load( const uint32_t frame )
{
    _impl->load( frame );
}

}
