/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *                          Daniel.Nachbaur@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "compartmentLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <brion/brion.h>
#include <brain/circuit.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

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

        for( size_t i = 0; i < morphologies.size(); ++i )
        {
            const brain::neuron::Morphology& morphology = *morphologies[i];

            // Adding the soma to the output.
            output.add( Event( morphology.getSoma().getCentroid(), 0.f ));
            // There's only one soma "section"
            const auto somaID =
                morphology.getSectionIDs({ brion::SECTION_SOMA })[0];
            _sections.push_back(
                SectionInfo( 1, _report.getOffsets()[i][somaID] ));

            const auto sections =
                morphology.getSectionIDs({ brion::SECTION_DENDRITE,
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
                    morphology.getSection( section ).getSamples( samples );
                for( const auto& point : points )
                    output.add( Event( point.get_sub_vector< 3 >(), 0.f ));

                _sections.push_back(
                    SectionInfo( compartments,
                                 _report.getOffsets()[i][section] ));

            }
        }
    }

    ssize_t load( const float time )
    {
        const brion::floatsPtr values = _report.loadFrame( time );
        if( !values )
            return -1;

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
        return index;
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

ssize_t CompartmentLoader::_load( const float time )
{
    return _impl->load( time );
}

}
