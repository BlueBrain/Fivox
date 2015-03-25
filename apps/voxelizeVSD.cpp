
/* Copyright (c) 2015, EPFL/Blue Brain Project
 *                     Stefan.Eilemann@epfl.ch
 */

#include <fivox/eventFunctor.h>
#include <fivox/imageSource.h>
#include <fivox/version.h>
#include <fivox/vsdLoader.h>
#include <fivox/eventFunctors/attenuationFunctor.h>
#include <itkImageFileWriter.h>
#include <lunchbox/file.h>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

// voxelizes the given BlueConfig (which requires an "area" and "soma" report
// into a volume.mhd/raw

namespace
{
    typedef itk::Image< uint8_t, 3 > Volume;
    typedef fivox::EventFunctor< Volume > Functor;
    typedef fivox::AttenuationFunctor< Volume > DyeFunctor;
    typedef fivox::SquaredDistanceFunctor< Volume > DefaultFunctor;
    typedef fivox::ImageSource< Volume, Functor > Filter;
}

using boost::lexical_cast;
namespace po = boost::program_options;

int main( int argc, char* argv[] )
{
    //----- Default values
    size_t size = 256;
    float time = 0.f;
    float cutOffDistance = 50.f;
    std::string outputFile( "voltageSensitiveDye.mhd" );
    std::string config( lunchbox::getExecutablePath() +
                        "/../share/Fivox/configs/BlueConfigVSD" );
    std::string target( "MiniColumn_0" );
    std::string dyeCurveFile;

    //----- Argument parsing
    po::variables_map vm;
    po::options_description desc( "Supported options" );
    desc.add_options()
        ( "help,h", "show help message." )
        ( "version,v", "Show program name and version." )
        ( "size,s", po::value< size_t >()->default_value( size ),
          "Size of the output volume" )
        ( "time,t", po::value< float >()->default_value( time ),
          "Timestep in voltage report to use" )
        ( "output,o", po::value< std::string >()->default_value( outputFile ),
          "Name of the output volume file" )
        ( "blueconfig,b", po::value< std::string >()->default_value( config ),
          "Name of the Blueconfig file" )
        ( "target,c", po::value< std::string >()->default_value( target ),
          "Name of the cell target" )
        ( "dyecurvefile,d", po::value< std::string >(),
          "The dye attenuation curve file to apply" )
        ( "cutoffdistance,f",
          po::value< float >()->default_value( cutOffDistance ),
          "The (micrometer) region of events considered per voxel" );

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

    if( vm.count( "size" ))
        size = vm["size"].as< size_t >();
    if( vm.count( "time" ))
        time = vm["time"].as< float >();
    if( vm.count( "output" ))
        outputFile = vm["output"].as< std::string >();
    if( vm.count( "blueconfig" ))
        config = vm["blueconfig"].as< std::string >();
    if( vm.count( "target" ))
        target = vm["target"].as< std::string >();
    if( vm.count( "dyecurvefile" ))
        dyeCurveFile = vm["dyecurvefile"].as< std::string >();
    if( vm.count( "cutoffdistance" ))
        cutOffDistance = vm["cutoffdistance"].as< float >();

    //----- Construct ITK pipeline
    typename Filter::Pointer filter = Filter::New();
    typename Volume::Pointer output = filter->GetOutput();
    typename Volume::SizeType vSize;
    typename Volume::RegionType region;

    vSize.Fill( size );
    region.SetSize( vSize );
    output->SetRegions( region );

    fivox::EventSourcePtr source = boost::make_shared< fivox::VSDLoader >(
                                       config, target, time );
    filter->GetFunctor().setSource( source );

    // set up size and origin for loaded circuit
    const fivox::AABBf& bbox = source->getBoundingBox();
    const float extent = bbox.getDimension().find_max();
    const float position = bbox.getMin().find_min();

    const float thickness = bbox.getDimension()[1];
    // Assuming that voxels are isotropic.
    const float circuitSpacePerVoxel = extent / float( size );

    boost::scoped_ptr< DyeFunctor > dyeFunctor;
    boost::scoped_ptr< DefaultFunctor > defaultFunctor;

    if( dyeCurveFile.empty())
    {
        defaultFunctor.reset( new DefaultFunctor( cutOffDistance ));

        filter->GetFunctor().setEventFunction(
                    boost::bind( &DefaultFunctor::fallOffFunction,
                                 defaultFunctor.get(), _1, _2 ));
    }
    else
    {
        dyeFunctor.reset( new DyeFunctor( dyeCurveFile,
                                          thickness,
                                          cutOffDistance ));

        filter->GetFunctor().setEventFunction(
                    boost::bind( &DyeFunctor::attenuationFunction,
                                 dyeFunctor.get(), _1, _2 ));
    }

    typename Volume::SpacingType spacing;
    spacing.Fill( circuitSpacePerVoxel );
    output->SetSpacing( spacing );

    typename Volume::PointType origin;
    origin.Fill( position );
    output->SetOrigin( origin );

    typedef itk::ImageFileWriter< Volume > Writer;
    typename Writer::Pointer writer = Writer::New();
    writer->SetInput( filter->GetOutput( ));
    writer->SetFileName( outputFile );

    writer->Update(); // run pipeline to write volume
}
