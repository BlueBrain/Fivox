
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

#include "beerLambertProjectionImageFilter.h"
#include "../commandLineApplication.h"
#include "../volumeHandler.h"
#include "../volumeWriter.h"

#include <itkImageFileWriter.h>

namespace
{
typedef float FloatPixelType;
typedef itk::Image< FloatPixelType, 2 > FloatImageType;
}

class ComputeVSD : public CommandLineApplication
{
public:
    ComputeVSD()
        : CommandLineApplication( "Compute the VSD signal and voxelize it to "
                                  "generate a 2D image through a Beer-Lambert "
                                  "projection (VTK format)" )
        , _outputFile( "image" )
    {
        _options.add_options()
//! [ComputeVSDParameters] @anchor ComputeVSD
            ( "output,o", po::value< std::string >(),
              "Name of the output volume file; contains frame number if "
              "--frames or --times" )
            ( "export-volume",
              "Export also the 3d volume containing the VSD values, "
              "in addition to the VTK file." )
            ( "sensor-res", po::value< size_t >()->default_value( 512 ),
              "Number of pixels per side of square sensor." )
            ( "sensor-dim", po::value< size_t >()->default_value( 1000 ),
              "Length of side of square sensor in micrometers." )
            ( "curve", po::value< std::string >(),
              "Path to the dye curve file (default: no file; attenuation of 1)")
            ( "depth", po::value< float >()->default_value( 2081.75641787f ),
              "Depth of the attenuation curve area of influence. It also "
              "defines the Y-coordinate at which it starts being applied, "
              "down until y=0 (default: 2081.75641787 micrometers)." )
            ( "interpolate-attenuation",
              "Interpolate the attenuation from the dye curve." )
            ( "sigma", po::value< double >()->default_value( 0.0045 ),
              "Absorption + scattering coefficient (units per micrometer) in "
              "the Beer-Lambert law. Must be a positive value "
              "(default: 0.0045).")
            ( "v0", po::value< float >()->default_value( -65.f ),
              "Resting potential (default: -65 mV." )
            ( "g0", po::value< float >()->default_value( 0.f ),
              "Multiplier for surface area in background fluorescence term." )
            ( "ap-threshold", po::value< float >(),
              "Action potential threshold in millivolts." );
//! [ComputeVSDParameters]
    }

    bool parse( int argc, char* argv[] ) final
    {
        if( !CommandLineApplication::parse( argc, argv ))
            return false;

        if( _vm.count( "output" ))
            _outputFile = _vm["output"].as< std::string >();

        return true;
    }

    /**
     * Write a floating point 2D image containing the Voltage-Sensitive Dye
     * projection, using a Beer-Lambert projection filter. The projection filter
     * computes the output using the real value of the data, i.e. not limited by
     * the precision of the final image
     *
     * @param input 3d volume used as input  to generate the 2D projection
     * @param filename name of the output image file
     */
    void projectVSD( VolumePtr input, const std::string& filename )
    {
        typedef BeerLambertProjectionImageFilter
            < fivox::FloatVolume, FloatImageType > FilterType;
        FilterType::Pointer projection = FilterType::New();
        projection->SetInput( input );
        projection->SetProjectionDimension( 1 ); // projection along Y-axis
        projection->SetPixelSize( input->GetSpacing( ).GetElement( 0 ));
        const double sigma = _vm["sigma"].as< double >();
        projection->SetSigma( sigma );

        // Write output image
        typedef itk::ImageFileWriter< FloatImageType > ImageWriter;
        ImageWriter::Pointer imageWriter = ImageWriter::New();
        imageWriter->SetInput( projection->GetOutput( ));

        const std::string& imageFile = filename + ".vtk";
        imageWriter->SetFileName( imageFile );
        imageWriter->Update();
        LBINFO << "VSD projection written as '" << imageFile
               << "' using a sigma value of " << sigma << std::endl;
    }

    void sample()
    {
        ::fivox::URI uri = getURI();

        const size_t sensorRes( _vm["sensor-res"].as< size_t >( ));
        const size_t sensorDim( _vm["sensor-dim"].as< size_t >( ));
        const float resolution = (float)sensorDim / sensorRes;
        // the URI handler takes voxels/unit as resolution
        uri.addQuery( "resolution", std::to_string( 1 / resolution ));

        const ::fivox::URIHandler params( uri );
        ImageSourcePtr source = params.newImageSource< fivox::FloatVolume >();

        ::fivox::EventSourcePtr loader = source->getEventSource();
        std::shared_ptr< fivox::VSDLoader > vsdLoader =
                std::static_pointer_cast< fivox::VSDLoader >( loader );

        const float v0 = _vm["v0"].as< float >();
        const float g0 = _vm["g0"].as< float >();

        LBINFO << "VSD info: v0 = " << v0 << " mV; g0 = " << g0 << std::endl;
        vsdLoader->setV0( v0 );
        vsdLoader->setG0( g0 );

        if( _vm.count( "ap-threshold" ))
        {
            vsdLoader->setSpikeFilter( true );
            const float apThreshold = _vm["ap-threshold"].as< float >();
            LBINFO << "Action potential threshold set to " << apThreshold
                   << " mV." << std::endl;
            vsdLoader->setApThreshold( apThreshold );
        }

        if( _vm.count( "curve" ))
        {
            const std::string& curveFile = _vm["curve"].as< std::string >();
            const float depth = _vm["depth"].as< float >();

            const bool interpolate = _vm.count( "interpolate-attenuation" );
            LBINFO << "Using '" << curveFile << "' as the dye curve file; "
                   << "depth of " << depth << " micrometers. "
                   << "Attenuation values will" << (!interpolate ? " not " :" ")
                   << "be interpolated." << std::endl;

            const fivox::AttenuationCurve dye( curveFile, depth );
            vsdLoader->setCurve( dye );
            vsdLoader->setInterpolation( interpolate );
        }

        const size_t size( std::ceil( source->getSizeInVoxel().find_max( )));

        // crop the volume region to the specified sensor dimensions
        fivox::Vector3f extent( source->getSizeInMicrometer( ));
        extent[0] = sensorDim;
        extent[2] = sensorDim;

        const VolumeHandler volumeHandler( size, extent );
        fivox::FloatVolume::IndexType vIndex;
        vIndex.Fill(0);
        fivox::FloatVolume::SizeType vSize;
        vSize[0] = extent[0] / resolution;
        vSize[1] = extent[1] / resolution;
        vSize[2] = extent[2] / resolution;

        VolumePtr output = source->GetOutput();
        output->SetRegions( fivox::FloatVolume::RegionType( vIndex, vSize ));
        output->SetSpacing( volumeHandler.computeSpacing( ));

        const fivox::AABBf& bbox = vsdLoader->getBoundingBoxSomas();
        output->SetOrigin( volumeHandler.computeOrigin( bbox.getCenter( )));

        VolumeWriter< float > writer( output, fivox::Vector2ui( ));

        const fivox::Vector2ui frameRange( getFrameRange( loader->getDt( )));
        const size_t numDigits = std::to_string( frameRange.y( )).length();
        for( uint32_t i = frameRange.x(); i < frameRange.y(); ++i )
        {
            std::string filename = _outputFile;
            if( frameRange.y() - frameRange.x() > 1 )
            {
                std::ostringstream os;
                os << filename << std::setfill('0') << std::setw(numDigits) << i;
                filename = os.str();
            }

            source->getEventSource()->setFrame( i );
            source->Modified();

            if( _vm.count( "export-volume" ))
            {
                const std::string& volumeName = filename + ".mhd";
                writer->SetFileName( volumeName );
                source->Modified();
                writer->Update(); // Run pipeline to write volume
                LBINFO << "Volume written as " << volumeName << std::endl;
            }

            projectVSD( output, filename );
        }
    }

private:
    std::string _outputFile;
};

int main( int argc, char* argv[] )
{
    ComputeVSD app;
    if( !app.parse( argc, argv ))
        return EXIT_SUCCESS;

    app.sample();
}
