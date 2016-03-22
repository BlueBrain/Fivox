/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
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

#include "uriHandler.h"

#include <fivox/compartmentLoader.h>
#include <fivox/densityFunctor.h>
#include <fivox/fieldFunctor.h>
#include <fivox/frequencyFunctor.h>
#ifdef FIVOX_USE_LFP
#  include <fivox/lfp/lfpFunctor.h>
#endif
#include <fivox/imageSource.h>
#include <fivox/somaLoader.h>
#include <fivox/spikeLoader.h>
#include <fivox/synapseLoader.h>
#include <fivox/testLoader.h>
#include <fivox/vsdLoader.h>
#ifdef FIVOX_USE_BBPTESTDATA
#  include <BBP/TestDatasets.h>
#endif
#include <lunchbox/file.h>
#include <lunchbox/log.h>
#include <lunchbox/uri.h>
#include <boost/lexical_cast.hpp>
#include <fivox/itk.h>

namespace fivox
{
namespace
{
using boost::lexical_cast;
const float _duration = 10.0f;
const float _dt = -1.0f; // loaders use experiment/report dt
const size_t _maxBlockSize = LB_64MB;
const float _resolution = 10.0f; // voxels per unit
const float _maxError = 0.001f;

EventSourcePtr _newLoader( const URIHandler& data )
{
    switch( data.getType( ))
    {
    case TYPE_COMPARTMENTS: return std::make_shared< CompartmentLoader >( data);
    case TYPE_SOMAS:        return std::make_shared< SomaLoader >( data );
    case TYPE_SPIKES:       return std::make_shared< SpikeLoader >( data );
    case TYPE_SYNAPSES:     return std::make_shared< SynapseLoader >( data );
    case TYPE_TEST:         return std::make_shared< TestLoader >( data );
    case TYPE_VSD:          return std::make_shared< VSDLoader >( data );
    default:                return nullptr;
    }
}

template< class T > std::shared_ptr< EventFunctor< itk::Image< T, 3 >>>
_newFunctor( const URIHandler& data )
{
    switch( data.getFunctorType( ))
    {
    case FUNCTOR_DENSITY:
        return std::make_shared< DensityFunctor< itk::Image< T, 3 >>>
                                                       ( data.getInputRange( ));
    case FUNCTOR_FIELD:
        return std::make_shared< FieldFunctor< itk::Image< T, 3 >>>
                                                       ( data.getInputRange( ));
    case FUNCTOR_FREQUENCY:
        return std::make_shared< FrequencyFunctor< itk::Image< T, 3 >>>
                                                       ( data.getInputRange( ));
#ifdef FIVOX_USE_LFP
    case FUNCTOR_LFP:
        return std::make_shared< LFPFunctor< itk::Image< T, 3 >>>
                                                       ( data.getInputRange( ));
#endif
    case FUNCTOR_UNKNOWN:
    default:
        return nullptr;
    }
}
}

class URIHandler::Impl
{
public:
    explicit Impl( const std::string& parameters )
        : uri( parameters )
        , config( uri.getPath( ))
        , target( _get( "target" ).empty() ? uri.getFragment() :
                                             _get( "target" ))
#ifdef FIVOX_USE_BBPTESTDATA
        , useTestData( config.empty( ))
#else
        , useTestData( false )
#endif
    {}

    std::string getConfig() const
    {
#ifdef FIVOX_USE_BBPTESTDATA
        if( useTestData )
        {
            return BBP_TEST_BLUECONFIG;
        }
#endif
        return config;
    }

    std::string getTarget( const std::string& defaultTarget ) const
    {
        if( target.empty( ))
        {
            if( defaultTarget.empty() && useTestData )
            {
                if( getType() == TYPE_SPIKES )
                    return "Column";
                return "mini50";
            }
            return defaultTarget;
        }
        return target;
    }

    std::string getReport() const
    {
        const std::string& report( _get( "report" ));
        if( report.empty( ))
        {
            switch( getType( ))
            {
            case TYPE_SOMAS:
                return "somas";
            default:
                return _get( "functor" ) == "lfp" ? "currents" : "voltages";
            }
        }
        return report;
    }

    float getDt() const { return _get( "dt", _dt ); }

    std::string getSpikes() const { return _get( "spikes" ); }

    float getDuration() const { return _get( "duration", _duration ); }

    Vector2f getInputRange() const
    {
        Vector2f defaultValue;
        switch( getType( ))
        {
        case TYPE_COMPARTMENTS:
            if( _get( "functor" ) == "lfp" )
                defaultValue = Vector2f( -1.47e-05f, 2.25e-03f );
            else
                defaultValue =
                        useTestData ? Vector2f( -190.f, 0.f )
                                    : Vector2f( brion::MINIMUM_VOLTAGE, 0.f );
            break;
        case TYPE_SOMAS:
            defaultValue =
                    useTestData ? Vector2f( -15.f, 0.f )
                                : Vector2f( brion::MINIMUM_VOLTAGE, 0.f );
            break;
        case TYPE_VSD:
            defaultValue = Vector2f( -100000.f, 300.f );
            break;
        case TYPE_SPIKES:
        case TYPE_SYNAPSES:
            defaultValue = Vector2f( 0.f, 2.f );
            break;
        default:
            defaultValue = Vector2f( 0.f, 10.f );
            break;
        }

        return Vector2f( _get( "inputMin", defaultValue[0] ),
                         _get( "inputMax", defaultValue[1] ));
    }

    std::string getDyeCurve() const { return _get( "dyecurve" ); }

    float getResolution() const { return _get( "resolution", _resolution ); }

    size_t getMaxBlockSize() const
        { return _get( "maxBlockSize", _maxBlockSize ); }

    float getMaxError() const
        { return std::max( _get( "maxError", _maxError ),
                           std::numeric_limits<float>::min( )); }

    bool showProgress() const
        { return _get( "showProgress", false ); }

    VolumeType getType() const
    {
        const std::string& scheme = uri.getScheme();
        if( scheme == "fivoxsomas" )
            return TYPE_SOMAS;
        if( scheme == "fivoxspikes" )
            return TYPE_SPIKES;
        if( scheme == "fivoxsynapses" )
            return TYPE_SYNAPSES;
        if( scheme == "fivoxvsd" )
            return TYPE_VSD;
        if( scheme == "fivox" || scheme == "fivoxcompartments" )
            return TYPE_COMPARTMENTS;
        if( scheme == "fivoxtest" )
            return TYPE_TEST;

        LBERROR << "Unknown URI scheme: " << scheme << std::endl;
        return TYPE_UNKNOWN;
    }

    FunctorType getFunctorType() const
    {
        const std::string& functor = _get( "functor" );
        if( functor == "density" )
            return FUNCTOR_DENSITY;
        if( functor == "lfp" )
            return FUNCTOR_LFP;
        if( functor == "field" )
            return FUNCTOR_FIELD;
        if( functor == "frequency" )
            return FUNCTOR_FREQUENCY;

        switch( getType( ))
        {
        case TYPE_SPIKES:
            return FUNCTOR_FREQUENCY;
        case TYPE_SYNAPSES:
            return FUNCTOR_DENSITY;
        case TYPE_COMPARTMENTS:
        case TYPE_SOMAS:
        case TYPE_VSD:
        case TYPE_TEST:
        default:
            return FUNCTOR_FIELD;
        }
    }

private:
    std::string _get( const std::string& param ) const
    {
        lunchbox::URI::ConstKVIter i = uri.findQuery( param );
        return i == uri.queryEnd() ? std::string() : i->second;
    }

    template< class T >
    T _get( const std::string& param, const T defaultValue ) const
    {
        const std::string& value = _get( param );
        if( value.empty( ))
            return defaultValue;

        try
        {
            return lexical_cast< T >( value );
        }
        catch( boost::bad_lexical_cast& )
        {
            LBWARN << "Invalid " << param << " specified, using "
                   << defaultValue << std::endl;
            return defaultValue;
        }
    }

    const lunchbox::URI uri;
    const std::string config;
    const std::string target;
    const bool useTestData;
};

URIHandler::URIHandler( const std::string& params )
    : _impl( new URIHandler::Impl( params ))
{}

URIHandler::~URIHandler()
{}

std::string URIHandler::getConfig() const
{
    return _impl->getConfig();
}

std::string URIHandler::getTarget( const std::string& defaultTarget ) const
{
    return _impl->getTarget( defaultTarget );
}

std::string URIHandler::getTarget() const
{
    return _impl->getTarget( std::string( ));
}

std::string URIHandler::getReport() const
{
    return _impl->getReport();
}

float URIHandler::getDt() const
{
    return _impl->getDt();
}

std::string URIHandler::getSpikes() const
{
    return _impl->getSpikes();
}

float URIHandler::getDuration() const
{
    return _impl->getDuration();
}

Vector2f URIHandler::getInputRange() const
{
    return _impl->getInputRange();
}

std::string URIHandler::getDyeCurve() const
{
    return _impl->getDyeCurve();
}

float URIHandler::getResolution() const
{
    return _impl->getResolution();
}

size_t URIHandler::getMaxBlockSize() const
{
    return _impl->getMaxBlockSize();
}

float URIHandler::getMaxError() const
{
    return _impl->getMaxError();
}

VolumeType URIHandler::getType() const
{
    return _impl->getType();
}

FunctorType URIHandler::getFunctorType() const
{
    return _impl->getFunctorType();
}

template< class T > itk::SmartPointer< ImageSource< itk::Image< T, 3 >>>
URIHandler::newImageSource() const
{
    LBINFO << "Loading events..." << std::endl;

    itk::SmartPointer< ImageSource< itk::Image< T, 3 >>> source =
        ImageSource< itk::Image< T, 3 >>::New();
    std::shared_ptr< EventFunctor< itk::Image< T, 3 >>> functor =
        _newFunctor< T >( *this );
    EventSourcePtr loader = _newLoader( *this );

    LBINFO << loader->getEvents().size() << " events " << *this << ", dt = "
           << loader->getDt() << " ready to voxelize" << std::endl;

    if( _impl->showProgress( ))
        source->showProgress();

    functor->setSource( loader );
    source->setFunctor( functor );
    return source;
}

std::ostream& operator << ( std::ostream& os, const URIHandler& params )
{
    switch( params.getType( ))
    {
    case TYPE_COMPARTMENTS:
        os << "compartment voltages from " << params.getReport();
        break;
    case TYPE_SOMAS:
        os << "soma voltages from " << params.getReport();
        break;
    case TYPE_SPIKES:
        os << "spikes from " << (params.getSpikes().empty() ? params.getConfig()
                                                           : params.getSpikes())
           << ", duration = " << params.getDuration();
        break;
    case TYPE_SYNAPSES:
        os << "synapse positions from " << params.getConfig();
        break;
    case TYPE_VSD:
        os << "VSD (Voltage-Sensitive Dye) from " << params.getReport();
        break;
    case TYPE_TEST:
        os << "test type for validation";
        break;
    case TYPE_UNKNOWN:
    default:
        os << "unknown data source " << params.getConfig();
        break;
    }

    os << ", using ";
    switch( params.getFunctorType( ))
    {
    case FUNCTOR_DENSITY:
        os << "density functor";
        break;
    case FUNCTOR_FIELD:
        os << "field functor";
        break;
    case FUNCTOR_FREQUENCY:
        os << "frequency functor";
        break;
    case FUNCTOR_LFP:
        os << "LFP functor";
        break;
    case FUNCTOR_UNKNOWN:
    default:
        os << "unknown functor";
        break;
    }

    return os << ", input data range = " << params.getInputRange()
              << ", resolution = " << params.getResolution();
}

}

// template instantiations
template fivox::ImageSource< itk::Image< uint8_t, 3 >>::Pointer
    fivox::URIHandler::newImageSource() const;
template fivox::ImageSource< itk::Image< float, 3 >>::Pointer
    fivox::URIHandler::newImageSource() const;
