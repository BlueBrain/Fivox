// Copyright (c) BBP/EPFL 2014
//                        Stefan.Eilemann@epfl.ch
// All rights reserved. Do not distribute without further notice.

// Voxelizes spheres into a volume

#include <fivox/version.h>
#include <itkMeanImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

typedef unsigned char               PixelType;
typedef itk::Image< PixelType, 3 >  ImageType;

int main( int argc, char* argv[] )
{
    // Default values
    size_t size = 32;
    std::string outputFile( "volume.mhd" );

    // Argument parsing
    po::variables_map vm;
    po::options_description desc( "Supported options" );
    desc.add_options()
        ( "help, h", "show help message." )
        ( "version, v", "Show program name and version." )
        ( "size,s", po::value< size_t >()->default_value( size ),
          "Size of the output volume" )
        ( "output,o", po::value< std::string >()->default_value( outputFile ),
          "Name of the output volume file" );
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
                  << std::endl
                  << "Copyright (c) BBP/EPFL 2014." << std::endl;
        return EXIT_SUCCESS;
    }

    if( vm.count( "size" ))
        size = vm["size"].as< size_t >();
    if( vm.count( "output" ))
        outputFile = vm["output"].as< std::string >();

    ImageType::SizeType vSize;
    vSize.Fill( size );

    ImageType::RegionType region;
    region.SetSize( vSize );

    ImageType::Pointer volume = ImageType::New();
    volume->SetRegions( region );
    volume->Allocate();

    typedef itk::MeanImageFilter< ImageType, ImageType >  FilterType;
    FilterType::Pointer meanFilter = FilterType::New();

    FilterType::InputSizeType radius;
    radius.Fill( 5.f );

    meanFilter->SetRadius( radius );
    meanFilter->SetInput( volume );

    typedef itk::ImageFileWriter< ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput( meanFilter->GetOutput() );
    writer->SetFileName( outputFile );

    try
    {
        writer->Update();
    }
    catch( itk::ExceptionObject & error )
    {
        std::cerr << "Error: " << error << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
