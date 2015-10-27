
/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Jafet.VillafrancaDiaz@epfl.ch
 *                     Stefan.Eilemann@epfl.ch
 */

#include <fivox/fivox.h>

#include <itkImageFileWriter.h>
#include <lunchbox/file.h>
#include <lunchbox/log.h>
#include <lunchbox/uri.h>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

namespace
{
typedef itk::Image< uint8_t, 3 > Volume;
typedef Volume::Pointer VolumePtr;
typedef std::shared_ptr< fivox::EventFunctor< Volume >> FunctorPtr;
typedef fivox::FieldFunctor< Volume > FieldFunctor;
typedef std::shared_ptr< FieldFunctor > FieldFunctorPtr;
typedef fivox::ImageSource< Volume > ImageSource;
typedef ImageSource::Pointer ImageSourcePtr;
}

using boost::lexical_cast;
namespace po = boost::program_options;

int main( int argc, char* argv[] )
{
    // Default values
    size_t size = 256;
    float time = 0.f;
    float cutOffDistance = 50.f;
    std::string outputFile( "volume.mhd" );
    std::string uri( "fivox://" );

    // Argument parsing
    po::variables_map vm;
    po::options_description desc( "Supported options", 140 /*line len*/ );
    desc.add_options()
        ( "help,h", "Show help message." )
        ( "version,v", "Show program name and version." )
        ( "volume", po::value< std::string >(),
//! [Usage]
          "Volume URI with parameters in the form:\n"
          "- Compartment reports:\n"
          "    fivox[compartments]://BlueConfig?report=string,target=string[ or #target]\n"
          "- Soma reports:\n"
          "    fivoxsomas://BlueConfig?report=string,target=string[ or #target]\n"
          "- Spike reports:\n"
          "    fivoxspikes://BlueConfig?duration=float,spikes=path,target=string[ or #target]\n"
          "- Synapse densities:\n"
          "    fivoxsynapses://BlueConfig?target=string[ or #target]\n"
          "- Voltage-sensitive dye reports:\n"
          "    fivoxvsd://BlueConfig?dyecurve=string,target=string[ or #target]\n"
          "\n"
          "Note: If target=string and #target parameters are given at the same time\n"
          "target=string has the precedence over #target parameter. Giving the #target as\n"
          "a parameter is deprecated\n"
          "\n"
          "Parameters for all types :\n"
          "- BlueConfig: BlueConfig file path\n"
          "              (default: 'configs/BlueConfigVSD' for VSD reports,\n"
          "               BBPTestData otherwise)\n"
          "- target: name of the BlueConfig target (default: CircuitTarget)\n"
          "- magnitude: value multiplied to each sampled voxel value\n"
          "             (defaults: 0.1 for Compartments and Somas,\n"
          "                        1.5 / duration for Spikes\n"
          "                        1.0 for Synapses and VSD)\n"
          "- functor: type of functor to sample the data into the voxels\n"
          "             (defaults: \"density\" for Synapses,\n"
          "                        \"frequency\" for Spikes,\n"
          "                        \"field\" for Compartments, Somas and VSD)\n"
          "- resolution: number of voxels per micrometer (default: 1.0)\n"
          "- maxBlockSize: maximum memory usage allowed for one block in bytes\n"
          "                (default: 64MB)\n"
          "\n"
          "Parameters for Compartments:\n"
          "- report: name of the compartment report\n"
          "          (default: 'voltage'; 'allvoltage' if BlueConfig is BBPTestData)\n"
          "- dt: timestep between requested frames in milliseconds (default: 10)\n"
          "\n"
          "Parameters for Somas:\n"
          "- report: name of the soma report\n"
          "          (default: 'soma'; 'voltage' if BlueConfig is BBPTestData)\n"
          "- dt: timestep between requested frames in milliseconds (default: 10)\n"
          "\n"
          "Parameters for Spikes:\n"
          "- duration: size of the time window to load spikes (default: dt)\n"
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
        ( "size,s", po::value< size_t >()->default_value( size ),
          "Size of the output volume." )
        ( "time,t", po::value< float >()->default_value( time ),
          "Timestep in voltage report to use." )
        ( "cutoffdistance,f",
          po::value< float >()->default_value( cutOffDistance ),
          "The (micrometer) region of events considered per voxel." )
        ( "output,o", po::value< std::string >()->default_value( outputFile ),
          "Name of the output volume file." );

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

    if( vm.count( "size" ))
        size = vm["size"].as< size_t >();
    if( vm.count( "time" ))
        time = vm["time"].as< float >();
    if( vm.count( "cutoffdistance" ))
        cutOffDistance = vm["cutoffdistance"].as< float >();
    if( vm.count( "output" ))
        outputFile = vm["output"].as< std::string >();

    ::fivox::URIHandler params( uri );

    ImageSourcePtr source = params.newImageSource< uint8_t >();
    FunctorPtr functor = source->getFunctor();
    FieldFunctorPtr fieldFunctor =
        std::dynamic_pointer_cast< FieldFunctor >( functor );
    if( fieldFunctor )
        fieldFunctor->setCutOffDistance( cutOffDistance );

    ::fivox::EventSourcePtr loader = functor->getSource();
    loader->load( time );

    Volume::SizeType vSize;
    vSize.Fill( size );

    Volume::RegionType region;
    region.SetSize( vSize );

    VolumePtr output = source->GetOutput();
    output->SetRegions( region );

    // Set up size and origin for loaded circuit
    const fivox::AABBf& bbox = loader->getBoundingBox();
    const fivox::Vector3f& position = bbox.getMin();
    const float extent = bbox.getDimension().find_max();

    typename Volume::SpacingType spacing;
    spacing.Fill( extent / float( size ));
    output->SetSpacing( spacing );

    typename Volume::PointType origin;
    origin[0] = position[0];
    origin[1] = position[1];
    origin[2] = position[2];
    output->SetOrigin( origin );

    typedef itk::ImageFileWriter< Volume > Writer;
    typename Writer::Pointer writer = Writer::New();
    writer->SetInput( output );
    writer->SetFileName( outputFile );

    writer->Update(); // Run pipeline to write volume
    LBINFO << "Volume written as " << outputFile << std::endl;
}
