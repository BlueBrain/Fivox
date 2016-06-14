
/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Stefan.Eilemann@epfl.ch
 *                          Daniel.Nachbaur@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Eyescale Software GmbH nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "volumeHandler.h"
#include "volumeWriter.h"

#include <fivox/fivox.h>

#include <lunchbox/file.h>
#include <lunchbox/log.h>
#include <lunchbox/uri.h>
#include <boost/program_options.hpp>

namespace
{
typedef fivox::FloatVolume::Pointer VolumePtr;
typedef fivox::ImageSource< fivox::FloatVolume > ImageSource;
typedef ImageSource::Pointer ImageSourcePtr;

template< typename T >
void _sample( ImageSourcePtr source, const vmml::Vector2ui& frameRange,
              const double sigmaVSDProjection, const fivox::URIHandler& params,
              const std::string& outputFile )
{
    VolumePtr input = source->GetOutput();
    VolumeWriter< T > writer( input, params.getInputRange( ));

    const size_t numDigits = std::to_string( frameRange.y( )).length();
    for( uint32_t i = frameRange.x(); i < frameRange.y(); ++i )
    {
        std::string filename;
        if( frameRange.y() - frameRange.x() > 1 )
        {
            std::ostringstream os;
            os << outputFile << std::setfill('0') << std::setw(numDigits) << i;
            filename = os.str();
        }
        else
            filename = outputFile;

        source->getFunctor()->getSource()->load( i );

        const std::string& volumeName = filename + ".mhd";
        writer->SetFileName( volumeName );
        source->Modified();
        writer->Update(); // Run pipeline to write volume
        LBINFO << "Volume written as " << volumeName << std::endl;

        if( sigmaVSDProjection < 0.0 )
            continue;

        writer.projectVSD( filename, sigmaVSDProjection );
    }
}
}

namespace vmml
{
std::istream& operator>>( std::istream& is, Vector2f& vec )
{
    return is >> std::skipws >> vec.x() >> vec.y();
}

std::istream& operator>>( std::istream& is, Vector2ui& vec )
{
    return is >> std::skipws >> vec.x() >> vec.y();
}
}

namespace po = boost::program_options;

int main( int argc, char* argv[] )
{
    // Default values
    size_t size = 256;
    std::string outputFile( "volume" );
    std::string uri( "fivox://" );

    // Argument parsing
    po::variables_map vm;
    po::options_description desc( "Supported options", 140 /*line len*/ );
    desc.add_options()
//! [Parameters] @anchor parameters
        ( "help,h", "Show help message" )
        ( "version,v", "Show program name and version" )
        ( "volume", po::value< std::string >(),
//! [Usage] @anchor voxelize
          "Volume URI with parameters in the form:\n"
          "- Compartment reports:\n"
          "    fivox[compartments]://BlueConfig?report=string&target=string[ or #target]\n"
          "- Soma reports:\n"
          "    fivoxsomas://BlueConfig?report=string&target=string[ or #target]\n"
          "- Spike reports:\n"
          "    fivoxspikes://BlueConfig?duration=float&spikes=path&target=string[ or #target]\n"
          "- Synapse densities:\n"
          "    fivoxsynapses://BlueConfig?target=string[ or #target]\n"
          "- Voltage-sensitive dye reports:\n"
          "    fivoxvsd://BlueConfig?dyecurve=string&target=string[ or #target]\n"
          "\n"
          "Note: If target=string and #target parameters are given at the same time\n"
          "target=string has the precedence over #target parameter. Giving the #target as\n"
          "a parameter is deprecated\n"
          "\n"
          "Parameters for all types :\n"
          "- BlueConfig: BlueConfig file path\n"
          "              (default: BBPTestData)\n"
          "- target: name of the BlueConfig target (default: CircuitTarget)\n"
          "- inputMin/inputMax: minimum and maximum input values to be considered for rescaling\n"
          "                     (defaults: [0.0, 2.0] for Spikes and Synapses\n"
          "                                [-190.0, 0.0] for Compartments with TestData, [-80.0, 0.0] otherwise\n"
          "                                [-15.0, 0.0] for Somas with TestData, [-80.0, 0.0] otherwise\n"
          "                                [-0.0000147, 0.00225] for LFP with TestData, [-10.0, 10.0] otherwise\n"
          "                                [-100000.0, 300.0] for VSD)\n"
          "- functor: type of functor to sample the data into the voxels\n"
          "             (defaults: \"density\" for Synapses,\n"
          "                        \"frequency\" for Spikes,\n"
          "                        \"field\" for Compartments, Somas and VSD)\n"
          "- resolution: number of voxels per micrometer (default: 10.0)\n"
          "- maxBlockSize: maximum memory usage allowed for one block in bytes\n"
          "                (default: 64MB)\n"
          "- cutoff: the cutoff distance in micrometers (default: 100).\n"
          "- showProgress: display progress bar for current voxelization step\n"
          "                (default: 0/off)\n"
          "\n"
          "Parameters for Compartments:\n"
          "- report: name of the compartment report\n"
          "          (default: 'voltage'; 'allvoltage' if BlueConfig is BBPTestData)\n"
          "- dt: timestep between requested frames in milliseconds\n"
          "      (default: report dt)\n"
          "\n"
          "Parameters for Somas:\n"
          "- report: name of the soma report\n"
          "          (default: 'soma'; 'voltage' if BlueConfig is BBPTestData)\n"
          "- dt: timestep between requested frames in milliseconds\n"
          "      (default: report dt)\n"
          "\n"
          "Parameters for Spikes:\n"
          "- duration: time window in milliseconds to load spikes (default: 10)\n"
          "- spikes: path to an alternate out.dat/out.spikes file\n"
          "          (default: SpikesPath specified in the BlueConfig)\n"
          "\n"
          "Parameters for VSD:\n"
          "- report: name of the soma report\n"
          "          (default: 'soma'; 'voltage' if BlueConfig is BBPTestData)\n"
          "- dyecurve: path to the dye curve file to apply, e.g. attenuation\n"
          "            (default: no file; attenuation of 1.0)\n"
//! [Usage]
          )
        ( "datatype,d", po::value< std::string >()->default_value( "float" ),
          "Type of the data in the output volume "
          "[float (default), int, short, char]" )
        ( "size,s", po::value< size_t >(),
          "Size of the output volume. If specified, this parameter will "
          "overwrite the resolution setting in the uri." )
        ( "time,t", po::value< float >(),
          "Timestamp to load in the report" )
        ( "times", po::value< fivox::Vector2f >(),
          "Time range [start end) to load in the report" )
        ( "frame,f", po::value< unsigned >(),
          "Frame to load in the report" )
        ( "frames", po::value< fivox::Vector2ui >(),
          "Frame range [start end) to load in the report" )
        ( "output,o", po::value< std::string >()->default_value( outputFile ),
          "Name of the output volume file (mhd and raw); contains frame number "
          "if --frames or --times" )
        ( "projection,p", po::value< double >(), "Generate the corresponding "
          "projected 2D image (only for VSD volumes), using the specified "
          "value as the absorption + scattering coefficient (units per "
          "micrometer) in the Beer-Lambert law. Must be a positive value." )
        ( "decompose", po::value< fivox::Vector2ui >(),
          "'rank size' data-decomposition for parallel job submission" );
//! [Parameters]

    po::store( po::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );

    if( vm.count( "help" ))
    {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }

    if( vm.count( "version" ))
    {
        std::cout << argv[0] << " version " << fivox::Version::getString()
                  << std::endl;
        return EXIT_SUCCESS;
    }

    if( vm.count( "volume" ))
        uri = vm["volume"].as< std::string >();
    else
        LBINFO << "Using " << uri << " as volume" << std::endl;

    if( vm.count( "output" ))
        outputFile = vm["output"].as< std::string >();

    fivox::Vector2ui decompose( 0, 1 );
    if( vm.count( "decompose" ))
    {
        decompose = vm["decompose"].as< fivox::Vector2ui >();

        const size_t numDigits = std::to_string( decompose[1] ).length();
        std::ostringstream os;
        os << "_" << std::setfill('0') << std::setw( numDigits )
           << decompose[0] << "_" << decompose[1];
        outputFile += os.str();
    }

    ::fivox::URIHandler params( uri );
    ImageSourcePtr source = params.newImageSource< float >();
    ::fivox::EventSourcePtr loader = source->getFunctor()->getSource();
    const fivox::AABBf& bbox = loader->getBoundingBox();

    if( vm.count( "size" ))
        size = vm["size"].as< size_t >();
    else
    {
        const fivox::Vector3f& sizeInVoxels = ( bbox.getSize() +
                                              loader->getCutOffDistance()
                                              * 2.f ) * params.getResolution();
        size = (size_t)std::ceil( sizeInVoxels.find_max( ));
    }

    const fivox::Vector3f& extent( bbox.getSize() +
                                   loader->getCutOffDistance() * 2.f );

    VolumeHandler volumeHandler( size, extent );

    VolumePtr output = source->GetOutput();
    output->SetRegions( volumeHandler.computeRegion( decompose ));
    output->SetSpacing( volumeHandler.computeSpacing( ));
    output->SetOrigin( volumeHandler.computeOrigin( bbox.getCenter( )));

    fivox::Vector2ui frameRange( 0, 1 ); // just frame 0 by default
    if( vm.count( "time" ))
    {
        const size_t frame = vm["time"].as< float >() / loader->getDt();
        frameRange = fivox::Vector2ui( frame, frame + 1 );
    }
    if( vm.count( "times" ))
    {
        const fivox::Vector2f times = vm["times"].as< fivox::Vector2f >();
        frameRange = fivox::Vector2ui( times.x() / loader->getDt(),
                                       times.y() / loader->getDt( ));
    }
    if( vm.count( "frame" ))
    {
        const size_t frame = vm["frame"].as< unsigned >();
        frameRange = fivox::Vector2ui( frame, frame + 1 );
    }
    if( vm.count( "frames" ))
        frameRange = vm["frames"].as< fivox::Vector2ui >();

    const double sigmaVSDProjection =
            params.getType() == fivox::TYPE_VSD && vm.count( "projection" ) ?
                vm["projection"].as< double >() : -1.0;

    const std::string& datatype( vm["datatype"].as< std::string >( ));
    if( datatype == "char" )
    {
        LBINFO << "Sampling volume as char (uint8_t) data" << std::endl;
        _sample< uint8_t >( source, frameRange, sigmaVSDProjection,
                            params, outputFile );
    }
    else if( datatype == "short" )
    {
        LBINFO << "Sampling volume as short (uint16_t) data" << std::endl;
        _sample< uint16_t >( source, frameRange, sigmaVSDProjection,
                             params, outputFile );
    }
    else if( datatype == "int" )
    {
        LBINFO << "Sampling volume as int (uint32_t) data" << std::endl;
        _sample< uint32_t >( source, frameRange, sigmaVSDProjection,
                             params, outputFile );
    }
    else
    {
        LBINFO << "Sampling volume as floating point data" << std::endl;
        _sample< float >( source, frameRange, sigmaVSDProjection,
                          params, outputFile );
    }
}
