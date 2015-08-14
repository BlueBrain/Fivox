
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
    po::options_description desc( "Supported options" );
    desc.add_options()
        ( "help,h", "show help message." )
        ( "version,v", "Show program name and version." )
        ( "volume", po::value< std::string >(),
//! [Usage]
          "Volume URI with parameters in the form:\n"
          "- Compartment reports:\n"
          "    fivox[compartments]:///BlueConfig?dt=float,report=string#target\n"
          "- Soma reports:\n"
          "    fivoxsomas:///BlueConfig?dt=float,report=string#target\n"
          "- Spike reports:\n"
          "    fivoxspikes:///BlueConfig?dt=float,duration=float,spikes=path#target\n"
          "- Synapse densities:\n"
          "    fivoxsynapses:///BlueConfig#target\n"
          "- Voltage-sensitive dye reports:\n"
          "    fivoxvsd:///BlueConfig?dt=float,dyecurve=string#target\n"
          "\n"
          "Parameters:\n"
          "- BlueConfig: BlueConfig file path\n"
          "              (default: 'configs/BlueConfigVSD' for VSD reports,\n"
          "               BBPTestData otherwise)\n"
          "- dt: duration in milliseconds of the timestep\n"
          "- duration: size of the time window to load spikes (default: 10)\n"
          "- target: name of the BlueConfig target\n"
          "- report: name of the compartment report\n"
          "- spikes: path to an alternate out.dat/out.spikes file\n"
          "          (default: SpikesPath specified in the BlueConfig)\n"
          "- dyecurve: path to the dye curve file to apply, e.g. attenuation\n"
          "            (only on VSD reports)\n"
          "- resolution: number of voxels per micrometer (default: 1.0)\n"
          "- maxBlockSize: maximum memory usage allowed for one block in bytes\n"
          "                (default: 16777216 bytes -- 16MB)\n"
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
