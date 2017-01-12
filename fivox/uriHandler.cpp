/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
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

#include "uriHandler.h"

#include <fivox/compartmentLoader.h>
#include <fivox/densityFunctor.h>
#include <fivox/fieldFunctor.h>
#include <fivox/frequencyFunctor.h>
#ifdef FIVOX_USE_LFP
#  include <fivox/lfp/lfpFunctor.h>
#endif
#ifdef FIVOX_USE_CUDA
#  include <fivox/cudaImageSource.h>
#endif
#include <fivox/eventValueSummationImageSource.h>
#include <fivox/functorImageSource.h>
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

#include <brion/blueConfig.h>
#include <brain/circuit.h>

namespace fivox
{
namespace
{
using boost::lexical_cast;
const float _duration = 10.0f;
const float _dt = -1.0f; // loaders use experiment/report dt
const size_t _maxBlockSize = LB_64MB;
const float _cutoff = 100.0f; // micrometers
const float _extend = 0.f; // micrometers
const float _gidFraction = 1.f;
}

class URIHandler::Impl
{
public:
    explicit Impl( const URI& parameters )
        : uri( parameters )
        , useTestData( false )
    {
        if( getType() == VolumeType::test )
            return;

#ifdef FIVOX_USE_BBPTESTDATA
        useTestData = uri.getPath().empty( );
        config.reset( new brion::BlueConfig( useTestData ? BBP_TEST_BLUECONFIG3
                                                         : uri.getPath( )));
#else
        config.reset( new brion::BlueConfig( uri.getPath( )));
#endif

        const brain::Circuit circuit( *config );
        const std::string target = _get( "target", _get( "postTarget",
                         useTestData ? "mini50" : config->getCircuitTarget( )));
        const std::string preTarget = _get( "preTarget" );
        const float gidFraction = getGIDFraction();
        if( target == "*" )
        {
            gids = gidFraction == 1.f ? circuit.getGIDs()
                                      : circuit.getRandomGIDs( gidFraction );
        }
        else
        {
            gids = gidFraction == 1.f ? circuit.getGIDs( target )
                                 : circuit.getRandomGIDs( gidFraction, target );

            if( !preTarget.empty( ))
            {
                preGIDs = gidFraction == 1.f ? circuit.getGIDs( preTarget )
                              : circuit.getRandomGIDs( gidFraction, preTarget );

                if( preGIDs.empty( ))
                    LBTHROW( std::runtime_error(
                             "No GIDs found for requested target '" +
                                 preTarget + "'" ));
            }
        }

        if( gids.empty( ))
            LBTHROW( std::runtime_error(
                     "No GIDs found for requested target '" + target + "'" ));
    }

    const brion::BlueConfig& getConfig() const
    {
        if( !config )
            LBTHROW( std::runtime_error(
                     "BlueConfig was not loaded" ));

        return *config;
    }

    const brion::GIDSet& getGIDs() const
    {
        if( !config )
            LBTHROW( std::runtime_error(
                     "BlueConfig was not loaded" ));

        return gids;
    }

    const brion::GIDSet& getPreGIDs() const
    {
        if( !config )
            LBTHROW( std::runtime_error(
                     "BlueConfig was not loaded" ));

        return preGIDs;
    }

    std::string getReport() const
    {
        const std::string& report( _get( "report" ));
        if( report.empty( ))
        {
            switch( getType( ))
            {
            case VolumeType::somas:
                return "somas";
            default:
                return _get( "functor" ) == "lfp" ? "currents" : "voltages";
            }
        }
        return report;
    }

    std::string getAreas() const
    {
        std::string areas = _get( "areas" );
        if( areas.empty() && getType() == VolumeType::vsd )
        {
#ifdef FIVOX_USE_BBPTESTDATA
            if( useTestData )
                return std::string( BBP_TESTDATA ) +
                     "/circuitBuilding_1000neurons/Neurodamus_output/areas.bbp";
#endif
            LBTHROW( std::runtime_error( "URI parameter 'areas' missing" ));
        }
        return areas;
    }

    float getDt() const { return _get( "dt", _dt ); }

    std::string getSpikes() const { return _get( "spikes" ); }

    float getDuration() const { return _get( "duration", _duration ); }

    Vector2f getInputRange() const
    {
        Vector2f defaultValue( brion::MINIMUM_VOLTAGE, 0.f );
        switch( getType( ))
        {
        case VolumeType::compartments:
            if( _get( "functor" ) == "lfp" )
                defaultValue = Vector2f( -1.47e-05f, 2.25e-03f );
            break;
        case VolumeType::somas:
            if( useTestData )
                defaultValue = Vector2f( -15.f, 0.f );
            break;
        case VolumeType::vsd:
            defaultValue = Vector2f( -100000.f, 300.f );
            break;
        case VolumeType::spikes:
            defaultValue = Vector2f( 0.f, 2.f );
            break;
        case VolumeType::synapses:
            defaultValue = FULLDATARANGE;
            break;
        default:
            defaultValue = Vector2f( 0.f, 10.f );
            break;
        }

        return Vector2f( _get( "inputMin", defaultValue[0] ),
                         _get( "inputMax", defaultValue[1] ));
    }

    float getResolution() const
    {
        float defaultResolution = 0.1f;
        switch( getType( ))
        {
        case VolumeType::spikes:
        case VolumeType::synapses:
            defaultResolution = 0.0625f; // 16 um/voxel
            break;
        default:
            break;
        }
        return _get( "resolution", defaultResolution );
    }

    size_t getMaxBlockSize() const
        { return _get( "maxBlockSize", _maxBlockSize ); }

    float getCutoffDistance() const
        { return std::max( _get( "cutoff", _cutoff ), 0.f ); }

    float getExtendDistance() const
        { return std::max( _get( "extend", _extend ), 0.f ); }

    float getGIDFraction() const
        { return _get( "gidFraction", _gidFraction ); }

    std::string getReferenceVolume() const
        { return _get( "reference" ); }

    size_t getSizeInVoxel() const
        { return _get( "size", 0 ); }

    std::string getDescription() const
    {
        std::stringstream desc;
        switch( getType( ))
        {
        case VolumeType::compartments:
        case VolumeType::somas:
            if( getFunctorType() == FunctorType::lfp )
                desc << "local field potential in mV of ";
            else
                desc << "membrane voltage in mV of ";
            if( getType() == VolumeType::somas )
                desc << "soma ";
            else
                desc << "compartment ";
            desc << "report '" << getReport() << "' for target '"
                 << _get( "target" ) << "'";
            break;
        case VolumeType::spikes:
            desc << "number of spikes in " << getDuration()
                 << "ms for target '" << _get( "target" ) << "'";
            break;
        case VolumeType::synapses:
        {
            desc << "number of synapses/voxel ";
            const std::string& preTarget = _get( "preTarget" );
            if( preTarget.empty( ))
                desc << "for afferent synapses of target '"
                     << _get( "target" ) << "'";
            else
                desc << "for pathway from '" << preTarget << "' to '"
                     << _get( "postTarget" ) << "'";
        } break;
        case VolumeType::vsd:
            desc << "VSD (Voltage-Sensitive Dye) from " << getReport()
                 << " for target '"  << _get( "target" ) << "'";
            break;
        case VolumeType::test:
        default:
            return "";
        }

        return desc.str();
    }

    VolumeType getType() const
    {
        const std::string& scheme = uri.getScheme();
        if( scheme == "fivoxsomas" )
            return VolumeType::somas;
        if( scheme == "fivoxspikes" )
            return VolumeType::spikes;
        if( scheme == "fivoxsynapses" )
            return VolumeType::synapses;
        if( scheme == "fivoxvsd" )
            return VolumeType::vsd;
        if( scheme == "fivox" || scheme == "fivoxcompartments" )
            return VolumeType::compartments;
        if( scheme == "fivoxtest" )
            return VolumeType::test;

        LBERROR << "Unknown URI scheme: " << scheme << std::endl;
        return VolumeType::unknown;
    }

    FunctorType getFunctorType() const
    {
        const std::string& functor = _get( "functor" );
        if( functor == "density" )
            return FunctorType::density;
        if( functor == "lfp" )
            return FunctorType::lfp;
        if( functor == "field" )
            return FunctorType::field;
        if( functor == "frequency" )
            return FunctorType::frequency;

        switch( getType( ))
        {
        case VolumeType::spikes:
        case VolumeType::synapses:
            LBTHROW( std::runtime_error(
                         "No functor support for synapses and spikes. "));
        case VolumeType::compartments:
        case VolumeType::somas:
        case VolumeType::vsd:
        case VolumeType::test:
        default:
            return FunctorType::field;
        }
    }

private:
    std::string _get( const std::string& param ) const
    {
        URI::ConstKVIter i = uri.findQuery( param );
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

    const URI uri;
    bool useTestData;
    std::unique_ptr< brion::BlueConfig> config;
    brion::GIDSet gids;
    brion::GIDSet preGIDs;
};

// bool specialization: param present with no value = true
template<> bool URIHandler::Impl::_get( const std::string& param,
                                        const bool defaultValue ) const
{
    URI::ConstKVIter i = uri.findQuery( param );
    if( i == uri.queryEnd( ))
        return defaultValue;
    if( i->second.empty( ))
        return true;

    try
    {
        return lexical_cast< bool >( i->second );
    }
    catch( boost::bad_lexical_cast& )
    {
        LBWARN << "Invalid " << param << " specified, using " << defaultValue
               << std::endl;
        return defaultValue;
    }
}

URIHandler::URIHandler( const URI& params )
    : _impl( new URIHandler::Impl( params ))
{}

URIHandler::~URIHandler()
{}

const brion::BlueConfig& URIHandler::getConfig() const
{
    return _impl->getConfig();
}

const brion::GIDSet& URIHandler::getGIDs() const
{
    return _impl->getGIDs();
}

const brion::GIDSet& URIHandler::getPreGIDs() const
{
    return _impl->getPreGIDs();
}

std::string URIHandler::getReport() const
{
    return _impl->getReport();
}

std::string URIHandler::getAreas() const
{
    return _impl->getAreas();
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

float URIHandler::getResolution() const
{
    return _impl->getResolution();
}

size_t URIHandler::getMaxBlockSize() const
{
    return _impl->getMaxBlockSize();
}

float URIHandler::getCutoffDistance() const
{
    return _impl->getCutoffDistance();
}

float URIHandler::getExtendDistance() const
{
    return _impl->getExtendDistance();
}

VolumeType URIHandler::getType() const
{
    return _impl->getType();
}

FunctorType URIHandler::getFunctorType() const
{
    return _impl->getFunctorType();
}

std::string URIHandler::getReferenceVolume() const
{
    return _impl->getReferenceVolume();
}

size_t URIHandler::getSizeInVoxel() const
{
    return _impl->getSizeInVoxel();
}

std::string URIHandler::getDescription() const
{
    return _impl->getDescription();
}

std::string URIHandler::getHelp()
{
    return
//! [VolumeParameters] @anchor VolumeParameters
        R"(- Compartment reports: fivox[compartments]://BlueConfig?report=string&target=string
- Soma reports: fivoxsomas://BlueConfig?report=string&target=string
- Spike reports: fivoxspikes://BlueConfig?duration=float&spikes=path&target=string
- Synapse densities: fivoxsynapses://BlueConfig?target=string
- Synapse densities for pathways: fivoxsynapses://BlueConfig?preTarget=string&postTarget=string
- Voltage-sensitive dye reports: fivoxvsd://BlueConfig?report=string&target=string

Parameters for all types :
- BlueConfig: BlueConfig file path (default: BBPTestData)
- target: name of the BlueConfig target (default: CircuitTarget)
- preTarget: target for presynaptic neurons for synapse densities (default: unset)
- postTarget: target for postsynaptic neurons for synapse densities (default: unset)
- gidFraction: take random cells from a fraction [0,1] of the given target (default: 1)
- inputMin/inputMax: minimum and maximum input values to be considered for rescaling
  (defaults: [0.0, 2.0] for Spikes
             [0.0, maxDensity] for Synapses
             [-80.0, 0.0] for Compartments
             [-15.0, 0.0] for Somas with TestData, [-80.0, 0.0] otherwise
             [-0.0000147, 0.00225] for LFP with TestData, [-10.0, 10.0] otherwise
             [-100000.0, 300.0] for VSD)
- functor: type of functor to sample the data into the voxels (defaults: 'density' for Synapses, 'frequency' for Spikes, 'field' for Compartments, Somas and VSD)
- maxBlockSize: maximum memory usage allowed for one block in bytes (default: 64MB)
- cutoff: the cutoff distance in micrometers (default: 100)
- extend: the additional distance, in micrometers, by which the original data volume will be extended in every dimension (default: 0, the volume extent matches the bounding box of the data events). Changing this parameter will result in more volumetric data, and therefore more computation time
- reference: path to a reference volume to take its size and resolution, overwrites the 'size' and 'resolution' parameter
- size: size in voxels along the largest dimension of the volume, overwrites the 'resolution' parameter
- resolution: number of voxels per micrometer (default: 0.0625 for densities, otherwise 0.1)

Parameters for Compartments:
- report: name of the compartment report (default: 'voltage'; 'allvoltage' if BlueConfig is BBPTestData)
- dt: timestep between requested frames in milliseconds (default: report dt)

Parameters for Somas:
- report: name of the soma report (default: 'soma'; 'voltage' if BlueConfig is BBPTestData)
- dt: timestep between requested frames in milliseconds (default: report dt)

Parameters for Spikes:
- duration: time window in milliseconds to load spikes (default: 10)
- spikes: path to an alternate out.dat/out.spikes file (default: SpikesPath specified in the BlueConfig)

Parameters for VSD:
- report: name of the voltage report (default: 'soma'; 'voltage' if BlueConfig is BBPTestData)
- areas: path to an area report file (default: path to TestData areas if BlueConfig is BBPTestData)
- dt: timestep between requested frames in milliseconds (default: report dt))";
//! [VolumeParameters]
}

template< class TImage >
ImageSourcePtr< TImage > URIHandler::newImageSource() const
{
    EventSourcePtr eventSource = newEventSource();

    ImageSourcePtr< TImage > source;
    switch( getType( ))
    {
    case VolumeType::spikes:
    case VolumeType::synapses:
    case VolumeType::vsd:
        source = EventValueSummationImageSource< TImage >::New();
        break;
    default:
#ifdef FIVOX_USE_CUDA
        bool cudaCapable = false;
        if( getFunctorType() == FunctorType::lfp )
        {
            int deviceCount = 0;
            if( cudaGetDeviceCount( &deviceCount ) != cudaSuccess )
                deviceCount = 0;

            cudaCapable = deviceCount > 0;
            if( cudaCapable )
            {
                LBINFO << "CUDA-capable device is detected. "
                       << "Using GPU implementation." << std::endl;
                source = CudaImageSource< TImage >::New();
            }
        }
        if( !cudaCapable )
#endif
        {
            LBINFO << "No CUDA-capable device is detected. "
                   << "Using CPU implementation." << std::endl;
            auto functorSource = FunctorImageSource< TImage >::New();
            auto functor = newFunctor< TImage >();
            functorSource->setFunctor( functor );
            functor->setEventSource( eventSource );
            source = functorSource;
        }
    }

    LBINFO << "Ready to voxelize " << *this << ", dt = "
           << eventSource->getDt() << std::endl;

    source->setEventSource( eventSource );
    source->setup( *this );
    return source;
}

EventSourcePtr URIHandler::newEventSource() const
{
    switch( getType( ))
    {
    case VolumeType::compartments:
        return std::make_shared< CompartmentLoader >( *this );
    case VolumeType::somas:
        return std::make_shared< SomaLoader >( *this );
    case VolumeType::spikes:
        return std::make_shared< SpikeLoader >( *this );
    case VolumeType::synapses:
        return std::make_shared< SynapseLoader >( *this );
    case VolumeType::test:
        return std::make_shared< TestLoader >( *this );
    case VolumeType::vsd:
        return std::make_shared< VSDLoader >( *this );
    default:
        return nullptr;
    }
}

template< class TImage >
EventFunctorPtr< TImage > URIHandler::newFunctor() const
{
    switch( getFunctorType( ))
    {
    case FunctorType::density:
        return std::make_shared< DensityFunctor< TImage >>();
    case FunctorType::field:
        return std::make_shared< FieldFunctor< TImage >>();
    case FunctorType::frequency:
        return std::make_shared< FrequencyFunctor< TImage >>();
#ifdef FIVOX_USE_LFP
    case FunctorType::lfp:
        return std::make_shared< LFPFunctor< TImage >>();
#endif
    case FunctorType::unknown:
    default:
        return nullptr;
    }
}

std::ostream& operator << ( std::ostream& os, const URIHandler& params )
{
    return os << params.getDescription() << ", resolution = "
              << 1.f/params.getResolution() << " um/voxel";
}

}

// template instantiations
template fivox::ImageSourcePtr< fivox::ByteVolume >
    fivox::URIHandler::newImageSource() const;
template fivox::ImageSourcePtr< fivox::FloatVolume >
    fivox::URIHandler::newImageSource() const;
template fivox::EventFunctorPtr< fivox::ByteVolume >
    fivox::URIHandler::newFunctor() const;
template fivox::EventFunctorPtr< fivox::FloatVolume >
    fivox::URIHandler::newFunctor() const;
