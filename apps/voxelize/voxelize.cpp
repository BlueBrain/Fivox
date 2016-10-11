
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

#include "../commandLineApplication.h"
#include "../volumeHandler.h"
#include "../volumeWriter.h"

namespace
{

template< typename T >
void _sample( ImageSourcePtr source, const vmml::Vector2ui& frameRange,
              const fivox::URIHandler& params, const std::string& outputFile )
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

        source->getEventSource()->setFrame( i );

        std::string volumeName = filename;
        if( volumeName.find_last_of( ".") == std::string::npos )
            volumeName += ".mhd";
        writer->SetFileName( volumeName );
        source->Modified();
        writer->Update(); // Run pipeline to write volume
        LBINFO << "Volume written as " << volumeName << std::endl;
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
              "Deprecated; use size in volume URI instead." )
            ( "output,o", po::value< std::string >(),
              "Name of the output volume file (mhd and raw); contains frame "
              "number if --frames or --times" )
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
        ::fivox::URI uri = getURI();

        // for compatibility
        if( _vm.count( "size" ))
            uri.addQuery( "size", std::to_string( _vm["size"].as< size_t >( )));

        const ::fivox::URIHandler params( uri );
        auto source = params.newImageSource< fivox::FloatVolume >();

        const fivox::Vector3f& extent( source->getSizeInMicrometer( ));
        const size_t size( std::ceil( source->getSizeInVoxel().find_max( )));

        const VolumeHandler volumeHandler( size, extent );
        VolumePtr output = source->GetOutput();

        output->SetRegions( volumeHandler.computeRegion( _decompose ));
        output->SetSpacing( volumeHandler.computeSpacing( ));
        const fivox::AABBf& bbox = source->getBoundingBox();
        output->SetOrigin( volumeHandler.computeOrigin( bbox.getCenter( )));

        ::fivox::EventSourcePtr loader = source->getEventSource();
        const fivox::Vector2ui frameRange( getFrameRange( loader->getDt( )));

        const std::string& datatype( _vm["datatype"].as< std::string >( ));
        if( datatype == "char" )
        {
            LBINFO << "Sampling volume as char (uint8_t) data" << std::endl;
            _sample< uint8_t >( source, frameRange, params, _outputFile );
        }
        else if( datatype == "short" )
        {
            LBINFO << "Sampling volume as short (uint16_t) data" << std::endl;
            _sample< uint16_t >( source, frameRange, params, _outputFile );
        }
        else if( datatype == "int" )
        {
            LBINFO << "Sampling volume as int (uint32_t) data" << std::endl;
            _sample< uint32_t >( source, frameRange, params, _outputFile );
        }
        else
        {
            LBINFO << "Sampling volume as floating point data" << std::endl;
            _sample< float >( source, frameRange, params, _outputFile );
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
