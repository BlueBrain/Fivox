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

int main( const int argc, const char* argv[] )
{
    // Argument parsing
    po::variables_map vm;
    po::options_description desc( "Supported options:" );
    desc.add_options()
        ( "help, h", "show help message." )
        ( "version, v", "Show program name and version." );
    po::store( parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );

    if( vm.count( "help" ))
    {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }

    if( vm.count( "version" ))
    {
        std::cout << "HelloRunner version " << fivox::Version::getString()
                  << std::endl
                  << "Copyright (c) BBP/EPFL 2014." << std::endl;
        return EXIT_SUCCESS;
    }

    const unsigned int Dimension = 2;

    const char * inputFileName = argv[1];
    const char * outputFileName = argv[2];
    const int radiusValue = atoi( argv[3] );

    typedef unsigned char                       PixelType;
    typedef itk::Image< PixelType, Dimension >  ImageType;

    typedef itk::ImageFileReader< ImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( inputFileName );

    typedef itk::MeanImageFilter< ImageType, ImageType >  FilterType;
    FilterType::Pointer meanFilter = FilterType::New();

    FilterType::InputSizeType radius;
    radius.Fill( radiusValue );

    meanFilter->SetRadius( radius );
    meanFilter->SetInput( reader->GetOutput() );

    typedef itk::ImageFileWriter< ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput( meanFilter->GetOutput() );
    writer->SetFileName( outputFileName );

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
