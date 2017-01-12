
/* Copyright (c) 2015-2017, EPFL/Blue Brain Project
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

#include <fivox/fivox.h>
#include <boost/program_options.hpp>
#include <lunchbox/string.h>
#include <lunchbox/term.h>

namespace po = boost::program_options;

namespace vmml
{
std::istream& operator >> ( std::istream& is, Vector3f& vec )
{
    return is >> std::skipws >> vec.x() >> vec.y() >> vec.z();
}

std::istream& operator >> ( std::istream& is, Vector2f& vec )
{
    return is >> std::skipws >> vec.x() >> vec.y();
}

std::istream& operator >> ( std::istream& is, Vector2ui& vec )
{
    return is >> std::skipws >> vec.x() >> vec.y();
}
}

namespace
{
typedef fivox::ImageSource< fivox::FloatVolume > ImageSource;
typedef ImageSource::Pointer ImageSourcePtr;

/**
 * Abstract class that behaves as a wrapper for the boost::program_options
 * evaluation, containing the common options for all Fivox applications:
 * volume URI and frame range options (time/times and frame/frames).
 *
 * Other applications can inherit from this class to extend the number of
 * options and functionality.
 */
class CommandLineApplication
{
public:
    CommandLineApplication( const std::string& caption )
        : _options( caption, lunchbox::term::getSize().first )
        , _uri( "fivox://" )
    {
        const std::string volumeHelp =
            std::string( "Volume URI with parameters in the form:\n" ) +
            lunchbox::string::prepend( fivox::URIHandler::getHelp(), "  " );
        _options.add_options()
//! [AppParameters] @anchor CommandLineApplication
            ( "help,h", "Show help message" )
            ( "version,v", "Show program name and version" )
            ( "volume", po::value< std::string >(), volumeHelp.c_str( ))
            ( "time,t", po::value< float >(),
              "Timestamp to load in the report" )
            ( "times", po::value< fivox::Vector2f >(),
              "Time range [start end) to load in the report" )
            ( "frame,f", po::value< unsigned >(),
              "Frame to load in the report" )
            ( "frames", po::value< fivox::Vector2ui >(),
              "Frame range [start end) to load in the report" );
//! [AppParameters]
    }

    virtual ~CommandLineApplication() {}

    /**
     * Parse the command line options, taking as parameters the count and list
     * of arguments (input for boost::program_options::parse_command_line)
     *
     * @return a flow control bool, returning false when the evaluation of the
     * options is not needed anymore (e.g. when using --help or --version), true
     * otherwise
     */
    virtual bool parse( int argc, char* argv[] )
    {
        po::store( po::parse_command_line( argc, argv, _options ), _vm );
        po::notify( _vm );

        if( _vm.count( "help" ))
        {
            std::cout << _options << std::endl;
            return false;
        }

        if( _vm.count( "version" ))
        {
            std::cout << argv[0] << " version " << fivox::Version::getString()
                      << std::endl;
            return false;
        }

        if( _vm.count( "volume" ))
            _uri = fivox::URI( _vm["volume"].as< std::string >( ));
        else
            LBINFO << "Using " << _uri << " as volume" << std::endl;

        return true;
    }

    /**
     * Return the frame range after evaluating the command line options,
     * considering also a dt parameter for the 'time' and 'times' options.
     *
     * @param dt the length of each frame, in microseconds. Needed to compute
     * the frame range when specifying only the 'time' or 'times' options
     * @return a vector of two components representing the frame range
     */
    fivox::Vector2ui getFrameRange( const float dt ) const
    {
        if( _vm.count( "time" ))
        {
            const size_t frame = _vm["time"].as< float >() / dt;
            return fivox::Vector2ui( frame, frame + 1 );
        }
        if( _vm.count( "times" ))
        {
            const fivox::Vector2f times = _vm["times"].as< fivox::Vector2f >();
            return fivox::Vector2ui( times.x() / dt, times.y() / dt );
        }
        if( _vm.count( "frame" ))
        {
            const size_t frame = _vm["frame"].as< unsigned >();
            return fivox::Vector2ui( frame, frame + 1 );
        }
        if( _vm.count( "frames" ))
            return _vm["frames"].as< fivox::Vector2ui >();

        return fivox::Vector2ui( 0, 1 ); // just frame 0 by default
    }

    /** @return the volume URI */
    const fivox::URI& getURI() const { return _uri; }

protected:
    po::options_description _options;
    po::variables_map _vm;
    fivox::URI _uri;
};

}
