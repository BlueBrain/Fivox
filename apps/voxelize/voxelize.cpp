
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
#include "../commandLineApplication.h"

namespace
{

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

class Voxelize : public CommandLineApplication
{
public:
    Voxelize()
        : CommandLineApplication( "Generate volumes for offline rendering in "
                                  "ParaView or other volume rendering "
                                  "applications" )
        , _outputFile( "volume" )
        , _decompose( 0, 1 )
    {
        _options.add_options()
//! [VoxelizeParameters] @anchor Voxelize
            ( "datatype,d", po::value< std::string >()->default_value("float"),
              "Type of the data in the output volume "
              "[float (default), int, short, char]" )
            ( "size,s", po::value< size_t >(),
              "Size of the output volume. If specified, this parameter will "
              "overwrite the resolution setting in the uri." )
            ( "output,o", po::value< std::string >(),
              "Name of the output volume file (mhd and raw); contains frame "
              "number if --frames or --times" )
            ( "projection,p", po::value< double >(), "Generate the "
              "corresponding projected 2D image (only for VSD volumes), using "
              "the specified value as the absorption + scattering coefficient "
              "(units per micrometer) in the Beer-Lambert law. Must be a "
              "positive value." )
            ( "decompose", po::value< fivox::Vector2ui >(),
              "'rank size' data-decomposition for parallel job submission" );
//! [VoxelizeParameters]
    }

    bool parse( int argc, char* argv[] ) final
    {
        if( !CommandLineApplication::parse( argc, argv ))
            return false;

        if( _vm.count( "output" ))
            _outputFile = _vm["output"].as< std::string >();

        if( _vm.count( "decompose" ))
        {
            _decompose = _vm["decompose"].as< fivox::Vector2ui >();

            const size_t numDigits = std::to_string( _decompose[1] ).length();
            std::ostringstream os;
            os << "_" << std::setfill('0') << std::setw( numDigits )
               << _decompose[0] << "_" << _decompose[1];
            _outputFile += os.str();
        }
        return true;
    }

    void sample()
    {
        const ::fivox::URIHandler params( getURI( ));
        ImageSourcePtr source = params.newImageSource< float >();
        ::fivox::EventSourcePtr loader = source->getFunctor()->getSource();
        const fivox::AABBf& bbox = loader->getBoundingBox();

        const fivox::Vector3f& extent( bbox.getSize() +
                                       params.getExtendDistance() * 2.f );

        size_t size;
        if( _vm.count( "size" ))
            size = _vm["size"].as< size_t >();
        else
        {
            const fivox::Vector3f& sizeVoxels = extent * params.getResolution();
            size = (size_t)std::ceil( sizeVoxels.find_max( ));
        }

        VolumeHandler volumeHandler( size, extent );

        VolumePtr output = source->GetOutput();
        output->SetRegions( volumeHandler.computeRegion( _decompose ));
        output->SetSpacing( volumeHandler.computeSpacing( ));
        output->SetOrigin( volumeHandler.computeOrigin( bbox.getCenter( )));

        const fivox::Vector2ui frameRange( getFrameRange( loader->getDt( )));

        const double sigmaVSDProjection =
            params.getType() == fivox::TYPE_VSD && _vm.count( "projection" ) ?
                                _vm["projection"].as< double >() : -1.0;

        const std::string& datatype( _vm["datatype"].as< std::string >( ));
        if( datatype == "char" )
        {
            LBINFO << "Sampling volume as char (uint8_t) data" << std::endl;
            _sample< uint8_t >( source, frameRange, sigmaVSDProjection,
                                params, _outputFile );
        }
        else if( datatype == "short" )
        {
            LBINFO << "Sampling volume as short (uint16_t) data" << std::endl;
            _sample< uint16_t >( source, frameRange, sigmaVSDProjection,
                                 params, _outputFile );
        }
        else if( datatype == "int" )
        {
            LBINFO << "Sampling volume as int (uint32_t) data" << std::endl;
            _sample< uint32_t >( source, frameRange, sigmaVSDProjection,
                                 params, _outputFile );
        }
        else
        {
            LBINFO << "Sampling volume as floating point data" << std::endl;
            _sample< float >( source, frameRange, sigmaVSDProjection,
                              params, _outputFile );
        }
    }

private:
    std::string _outputFile;
    ::fivox::Vector2ui _decompose;
};

int main( int argc, char* argv[] )
{
    Voxelize app;
    if( !app.parse( argc, argv ))
        return EXIT_SUCCESS;

    app.sample();
}
