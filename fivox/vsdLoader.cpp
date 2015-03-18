/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include "vsdLoader.h"
#include "event.h"

#include <BBP/BBP.h>
#include <boost/foreach.hpp>

namespace fivox
{

namespace detail
{
class VSDLoader
{
public:
    VSDLoader( fivox::EventSource& output, const std::string& blueconfig,
               const std::string& target, const float time )
        : _output( output )
        , _experiment( blueconfig )
        , _target( _experiment.cell_target( target ))
        , _voltages( *_experiment.reports().find( "soma" ),
                     _experiment.cell_target( target ))
        , _areas( *_experiment.reports().find( "area" ),
                  _experiment.cell_target( "Mosaic" ))
    {
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( _target, bbp::NEURONS | bbp::MORPHOLOGIES );
        _areas.updateMapping( _target );

        if( !_areas.loadFrame( 0.f, _areasFrame ))
            throw std::runtime_error( "Can't load 'area' vsd report" );

        size_t i=0;
        BOOST_FOREACH( const uint32_t gid, _target )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            const bbp::Sections& sections = neuron.dendrites();
            size_t j = 0;
            BOOST_FOREACH( const bbp::Section& section, sections )
            {
                const size_t nCompartments =_areas.getCompartmentCounts()[i][j];
                const float compartmentLength = 1.f / float( nCompartments );
                const float compartmentMiddle = compartmentLength * .5f;

                for( size_t k = 0; k < nCompartments; ++k )
                {
                    const bbp::Cross_Section& midPoint = section.cross_section(
                        compartmentMiddle + k * compartmentLength );

                    output.add( Event( midPoint.center(), 0.f ));
                }
                ++j;
            }
            ++i;
        }

        LBCHECK( loadFrame( time ));
    }

    bool loadFrame( const float time )
    {
        bbp::CompartmentReportFrame voltages;
        if( !_voltages.loadFrame( time, voltages ))
        {
            std::cerr << "Could not load frame at " << time << "ms" <<std::endl;
            return false;
        }

        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.update( voltages );
        const bbp::floatsPtr& areas = _areasFrame.getData< bbp::floatsPtr >();

        size_t i=0;
        size_t index = 0;
        BOOST_FOREACH( const uint32_t gid, _target )
        {
            const bbp::Neuron& neuron = microcircuit.neuron( gid );
            const bbp::Sections& sections = neuron.dendrites();

            size_t j = 0;
            BOOST_FOREACH( const bbp::Section& section, sections )
            {
                const size_t nCompartments =_areas.getCompartmentCounts()[i][j];
                uint64_t offset = _areas.getOffsets()[i][j];

                for( size_t k = 0; k < nCompartments; ++k )
                    _output.update( index++,
                                    neuron.voltage() * (*areas)[ offset++ ] );
                ++j;
                LBVERB << section.id() << std::endl;
            }
            ++i;
        }
        return true;
    }

private:
    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    const bbp::Cell_Target _target;
    bbp::CompartmentReportReader _voltages;
    bbp::CompartmentReportReader _areas;
    bbp::CompartmentReportFrame _areasFrame;
};
}

VSDLoader::VSDLoader( const std::string& blueconfig, const std::string& target,
                      const float time )
    : _impl( new detail::VSDLoader( *this, blueconfig, target, time ))
{}

VSDLoader::~VSDLoader()
{
    delete _impl;
}

bool VSDLoader::loadFrame( const float time )
{
    return _impl->loadFrame( time );
}

}
