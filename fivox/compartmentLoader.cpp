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
        , _currentTime( -1.f )
        , _dt( params.getDt( ))
        , _magnitude( params.getMagnitude( ))
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

            for( uint64_t offset = info.offset; offset < end; ++offset )
            {
                _output.update( index++, _magnitude * (( *voltages )[ offset ] -
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

        const float time  = _reader.getStartTime() + _dt * frame;
        return load( time );
    }

    Vector2ui getFrameRange()
    {
        const float reportTime = _reader.getEndTime() - _reader.getStartTime();
        const uint32_t numFrames = std::ceil( reportTime / _dt );
        return Vector2ui( 0, numFrames );
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;
    SectionInfos _sections;

    float _currentTime;
    float _dt;
    const float _magnitude;
};

CompartmentLoader::CompartmentLoader( const URIHandler& params )
    : _impl( new CompartmentLoader::Impl( *this, params ))
{}

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
