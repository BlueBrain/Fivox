/* Copyright (c) 2014-2015, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 */

#include "somaLoader.h"
#include "event.h"
#include "uriHandler.h"

#include <BBP/BBP.h>

#ifdef final
#  undef final
#endif

namespace fivox
{
class SomaLoader::Impl
{
public:
    Impl( fivox::EventSource& output, const URIHandler& params )
        : _output( output )
        , _experiment( params.getConfig( ))
        , _reader( *_experiment.reports().find( params.getReport( )),
                   _experiment.cell_target(
                       params.getTarget( _experiment.circuit_target( ))))
    {
        bbp::Microcircuit& microcircuit = _experiment.microcircuit();
        microcircuit.load( _reader.getCellTarget(), bbp::NEURONS );

        const bbp::Neurons& neurons = microcircuit.neurons();
        for( const bbp::Neuron& neuron : neurons )
            output.add( Event( neuron.position(), 0.f ));
    }

    bool load( const float time )
    {
        bbp::CompartmentReportFrame frame;
        if( !_reader.loadFrame( time, frame ))
        {
            LBERROR << "Could not load frame at " << time << "ms" << std::endl;
            return false;
        }

        _experiment.microcircuit().update( frame );
        const bbp::Neurons& neurons = _experiment.microcircuit().neurons();
        size_t i = 0;
        for( const bbp::Neuron& neuron : neurons )
            _output.update( i++, neuron.voltage() - brion::MINIMUM_VOLTAGE );
        return true;
    }

    fivox::EventSource& _output;
    bbp::Experiment _experiment;
    bbp::CompartmentReportReader _reader;
};

SomaLoader::SomaLoader( const URIHandler& params )
    : EventSource( params )
    , _impl( new SomaLoader::Impl( *this, params ))
{
    if( getDt() < 0.f )
        setDt( _impl->_reader.getTimestep( ));
}

SomaLoader::~SomaLoader()
{}

Vector2f SomaLoader::_getTimeRange() const
{
    return Vector2f( _impl->_reader.getStartTime(),
                     _impl->_reader.getEndTime( ));
}

bool SomaLoader::_load( const float time )
{
    return _impl->load( time );
}

}
