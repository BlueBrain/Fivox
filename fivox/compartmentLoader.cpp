/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *                          Daniel.Nachbaur@epfl.ch
 */

#include "compartmentLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <brion/brion.h>
#include <brain/circuit.h>
#include <brain/morphology.h>

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
        , _config( params.getConfig( ))
        , _target( _config.parseTarget( params.getTarget( )))
        , _report( _config.getReportSource( params.getReport( )),
                   brion::MODE_READ, _target)
    {
        _report.updateMapping( _target );

        brain::Circuit circuit( _config );
        const auto morphologies = circuit.loadMorphologies(
            _target, brain::Circuit::COORDINATES_GLOBAL );

        for( size_t i = 0; i != morphologies.size(); ++i )
        {
            const brain::Morphology& morphology = *morphologies[i];

            // The voltage reports typically include the soma, dendrites and
            // first two axon sections. Here we will ignore the axon sections.
            const auto sections =
                morphology.getSectionIDs({ brion::SECTION_SOMA,
                                           brion::SECTION_DENDRITE,
                                           brion::SECTION_APICAL_DENDRITE });
            for( auto section : sections )
            {
                const auto& counts = _report.getCompartmentCounts()[i];
                if( section >= counts.size( ))
                    continue;
                const size_t compartments = counts[section];
                if( !compartments )
                    continue; // This occurs in unreported axonal sections.

                brion::floats samples;
                samples.reserve( compartments );
                const float length = 1.f / float( compartments );
                for( float k = length * .5f; k < 1.0; k += length )
                    samples.push_back( k );

                const auto points =
                    morphology.getSectionSamples( section, samples );
                for( const auto& point : points )
                    output.add( Event( point.get_sub_vector< 3 >(), 0.f ));

                _sections.push_back(
                    SectionInfo( compartments,
                                 _report.getOffsets()[i][section] ));

            }
        }
    }

    bool load( const float time )
    {
        const brion::floatsPtr values = _report.loadFrame( time );
        if( !values )
        {
            LBERROR << "Could not load frame at " << time << "ms" << std::endl;
            return false;
        }

        size_t index = 0;
        for( const auto& section : _sections )
        {
            const uint64_t end = section.numCompartments + section.offset;
            for( uint64_t offset = section.offset; offset < end;
                 ++offset, ++index )
            {
                _output.update(
                    index, (( *values )[ offset ] - brion::MINIMUM_VOLTAGE ));
            }
        }
        LBINFO << "Updated " << index << " events at " << time << "ms"
               << std::endl;
        return true;
    }

    fivox::EventSource& _output;

    brion::BlueConfig _config;
    brion::GIDSet _target;
    SectionInfos _sections;

    brion::CompartmentReport _report;
};

CompartmentLoader::CompartmentLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new CompartmentLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_report.getTimestep( ));
}

CompartmentLoader::~CompartmentLoader()
{}

Vector2f CompartmentLoader::_getTimeRange() const
{
    return Vector2f( _impl->_report.getStartTime(),
                     _impl->_report.getEndTime( ));
}

bool CompartmentLoader::_load( const float time )
{
    return _impl->load( time );
}

}
