
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

#include "../commandLineApplication.h"
#include "../volumeWriter.h"
#include "projectionImageFilter.h"

#include <itkImageFileWriter.h>

namespace
{
typedef float FloatPixelType;
typedef itk::Image<FloatPixelType, 2> FloatImageType;
}

class ComputeVSD : public CommandLineApplication
{
public:
    ComputeVSD()
        : CommandLineApplication(
              "A tool to compute 2D projections of VSD "
              "signals using the Beer-Lambert law.",
              "fivoxvsd://")
        , _outputFile("image")
    {
        // clang-format off
        _options.add_options()
//! [ComputeVSDParameters] @anchor ComputeVSD
            ("output,o", po::value<std::string>(),
             "Name of the output volume (if 'export-volume' specified) and 2D "
             "image (VTK format) files; contains frame number if --frames or "
             "--times specified.")
            ("export-volume",
             "Export also the 3d volume (mhd + raw) containing the VSD values,"
             " in addition to the VTK file.")
            ("export-point-sprites",
             "Export also the point sprite files containing the VSD values,"
             " in addition to the VTK file.")
            ("sensor-res", po::value<size_t>()->default_value(512),
             "Number of pixels per side of square sensor.")
            ("sensor-dim", po::value<size_t>()->default_value(1000),
             "Length of side of square sensor in micrometers.")
            ("curve", po::value<std::string>(),
             "Path to the dye curve file (default: no attenuation)")
            ("depth", po::value<float>()->default_value(2081.75641787f),
             "Depth of the attenuation curve area of influence. It also "
             "defines the Y-coordinate at which it starts being applied, "
             "down until y=0 (default: 2081.75641787 micrometers).")
            ("interpolate-attenuation",
             "Interpolate the attenuation from the dye curve.")
            ("sigma", po::value<double>()->default_value(0.0045, "0.0045"),
             "Absorption + scattering coefficient (units per micrometer) in "
             "the Beer-Lambert law. Must be a positive value "
             "(default: 0.0045).")
            ("v0", po::value<float>()->default_value(-65.f),
             "Resting potential (default: -65 mV).")
            ("g0", po::value<float>()->default_value(0.f),
             "Multiplier for surface area in background fluorescence term.")
            ("ap-threshold", po::value<float>(),
             "Action potential threshold in millivolts.")
            ("soma-pixels", po::value<std::string>(),
             "Produce a text file containing the GIDs loaded and their "
             "corresponding 3D positions and indices in the resulting 2D "
             "image.");
//! [ComputeVSDParameters]
        // clang-format on
    }

    bool parse(int argc, char* argv[]) final
    {
        if (!CommandLineApplication::parse(argc, argv))
            return false;

        if (_vm.count("output"))
            _outputFile = _vm["output"].as<std::string>();

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
    void projectVSD(VolumePtr input, const std::string& filename)
    {
        typedef ProjectionImageFilter<fivox::FloatVolume, FloatImageType>
            FilterType;
        FilterType::Pointer projection = FilterType::New();
        projection->SetInput(input);
        projection->SetProjectionDimension(1); // projection along Y-axis

        // Write output image
        typedef itk::ImageFileWriter<FloatImageType> ImageWriter;
        ImageWriter::Pointer imageWriter = ImageWriter::New();
        imageWriter->SetInput(projection->GetOutput());

        const std::string& imageFile = filename + ".vtk";
        imageWriter->SetFileName(imageFile);
        imageWriter->Update();
    }

    void sample()
    {
        ::fivox::URI uri = getURI();

        const size_t sensorRes(_vm["sensor-res"].as<size_t>());
        const size_t sensorDim(_vm["sensor-dim"].as<size_t>());
        const float resolution = (float)sensorDim / sensorRes;
        // the URI handler takes voxels/unit as resolution
        uri.addQuery("resolution", std::to_string(1 / resolution));

        const ::fivox::URIHandler params(uri);
        ImageSourcePtr source = params.newImageSource<fivox::FloatVolume>();

        _eventSource = source->getEventSource();
        std::shared_ptr<fivox::VSDLoader> vsdLoader =
            std::static_pointer_cast<fivox::VSDLoader>(_eventSource);

        const float v0 = _vm["v0"].as<float>();
        const float g0 = _vm["g0"].as<float>();

        LBINFO << "VSD info: V0 = " << v0 << " mV; G0 = " << g0 << std::endl;
        vsdLoader->setRestingPotential(v0);
        vsdLoader->setAreaMultiplier(g0);
        const double sigma = _vm["sigma"].as<double>();
        vsdLoader->setSigma(sigma);

        if (_vm.count("ap-threshold"))
        {
            vsdLoader->setSpikeFilter(true);
            const float apThreshold = _vm["ap-threshold"].as<float>();
            LBINFO << "Action potential threshold set to " << apThreshold
                   << " mV." << std::endl;
            vsdLoader->setApThreshold(apThreshold);
        }

        if (_vm.count("curve"))
        {
            const std::string& curveFile = _vm["curve"].as<std::string>();
            const float depth = _vm["depth"].as<float>();

            const bool interpolate = _vm.count("interpolate-attenuation");
            LBINFO << "Using '" << curveFile << "' as the dye curve file; "
                   << "depth of " << depth << " micrometers. "
                   << "Attenuation values will"
                   << (!interpolate ? " not " : " ") << "be interpolated."
                   << std::endl;

            const fivox::AttenuationCurve dye(curveFile, depth);
            vsdLoader->setCurve(dye);
            vsdLoader->setInterpolation(interpolate);
        }

        const size_t size(source->getSizeInVoxel().find_max());

        // crop the volume region to the specified sensor dimensions
        fivox::Vector3f extent(source->getSizeInMicrometer());
        extent[0] = sensorDim;
        extent[2] = sensorDim;

        const fivox::VolumeHandler volumeHandler(size, extent);
        fivox::FloatVolume::IndexType vIndex;
        vIndex.Fill(0);
        fivox::FloatVolume::SizeType vSize;
        vSize[0] = extent[0] / resolution;
        vSize[1] = extent[1] / resolution;
        vSize[2] = extent[2] / resolution;

        VolumePtr output = source->GetOutput();
        output->SetRegions(fivox::FloatVolume::RegionType(vIndex, vSize));

        fivox::AABBf bboxSomas;
        const auto& somaPositions = vsdLoader->getSomaPositions();
        for (const auto& position : somaPositions)
            bboxSomas.merge(position);

        // pixel/voxel size
        const auto spacing = extent[1] / source->getSizeInVoxel()[1];
        // left bottom corner of the image/volume
        const auto origin = volumeHandler.computeOrigin(bboxSomas.getCenter());

        if (_vm.count("soma-pixels"))
        {
            const auto& fileName = _vm["soma-pixels"].as<std::string>();
            std::ofstream file(fileName);
            if (!file.is_open())
                LBERROR << "File " << fileName << " could not be opened"
                        << std::endl;
            else
            {
                file << "# Soma position and corresponding pixel index for "
                        "each cell, in the following format:\n"
                     << "#     gid [ posX posY posZ ]: i j\n"
                     << "# File version: 1\n"
                     << "# Fivox version: " << fivox::Version::getString()
                     << std::endl;

                size_t i = 0;
                const auto& gids = vsdLoader->getGIDs();
                for (const auto& gid : gids)
                {
                    if (file.bad())
                        break;
                    const auto& pos = somaPositions[i++];
                    file << gid << " " << pos << ": "
                         << std::floor((pos[0] - origin[0]) / spacing) << " "
                         << std::floor((pos[2] - origin[2]) / spacing)
                         << std::endl;
                }
            }
            if (file.good())
                LBINFO << "Soma positions written as " << fileName << std::endl;
            else
                LBERROR << "Error while writing to " << fileName << std::endl;
        }

        output->SetSpacing(spacing);
        output->SetOrigin(origin);

        vsdLoader->setYOrigin(output->GetOrigin()[1]);
        vsdLoader->setCircuitHeight(_vm["depth"].as<float>());

        VolumeWriter<float> writer(output, fivox::Vector2ui());

        const fivox::Vector2ui frameRange(getFrameRange(_eventSource->getDt()));
        size_t numDigits = std::to_string(frameRange.y()).length();
        if (_vm.count("times"))
        {
            const float endTime = _vm["times"].as<fivox::Vector2f>()[1];
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << endTime;
            numDigits = s.str().length();
        }

        if (_vm.count("export-point-sprites"))
            _writePointSpritePositions();

        for (uint32_t i = frameRange.x(); i < frameRange.y(); ++i)
        {
            std::string filename = _outputFile;
            if (frameRange.y() - frameRange.x() > 1)
            {
                // append the frame number if --frames, timestamp otherwise
                std::ostringstream os;
                os << filename << std::setfill('0') << std::setw(numDigits);
                if (_vm.count("times"))
                    os << std::fixed << std::setprecision(1)
                       << i * vsdLoader->getDt();
                else
                    os << i;

                filename = os.str();
            }

            _eventSource->setFrame(i);
            source->Modified();

            if (_vm.count("export-volume"))
            {
                const std::string& volumeName = filename + ".mhd";
                writer->SetFileName(volumeName);
                source->Modified();
                writer->Update(); // Run pipeline to write volume
                LBINFO << "Volume written as " << volumeName << std::endl;
            }

            projectVSD(output, filename);

            if (_vm.count("export-point-sprites"))
            {
                _writePointSpriteIntensities(filename);
                _writePointSpriteHeader(filename);
            }
        }
    }

private:
    void _writePointSpriteHeader(const std::string& filename) const
    {
        const std::string& pshFile(filename + ".psh");
        std::ofstream file(pshFile.c_str());
        if (file.is_open())
        {
            file << "# VSD Point Sprite files\n"
                 << "# File version: 1\n"
                 << "# Fivox version: " << fivox::Version::getString()
                 << std::endl;

            file << "EventsCount=" << _eventSource->getNumEvents() << std::endl;
            const fivox::AABBf& bbox(_eventSource->getBoundingBox());
            file << "XCenter=" << bbox.getCenter().x() << std::endl;
            file << "YCenter=" << bbox.getCenter().y() << std::endl;
            file << "ZCenter=" << bbox.getCenter().z() << std::endl;
            file << "AABBWidth=" << bbox.getSize().x() << std::endl;
            file << "AABBHeight=" << bbox.getSize().y() << std::endl;
            file << "AABBDepth=" << bbox.getSize().z() << std::endl;

            file << "VSDPositionFile=" << _outputFile + ".psp" << std::endl;
            file << "VSDIntensityFile=" << filename + ".psi" << std::endl;

            // Should be called TimeStamp but need to remain TimeStep because of
            // compatibility issues.
            file << "TimeStep=" << _eventSource->getCurrentTime() << std::endl;

            if (file.good())
                LBINFO << "Point Sprite header written as " << pshFile
                       << std::endl;
        }
    }

    void _writePointSpritePositions() const
    {
        const std::string& pspFile(_outputFile + ".psp");
        std::ofstream file(pspFile.c_str(), std::ios::binary);
        if (file.is_open())
        {
            for (size_t i = 0; i < _eventSource->getNumEvents(); ++i)
            {
                file.write((const char*)&_eventSource->getPositionsX()[i],
                           sizeof(float));
                file.write((const char*)&_eventSource->getPositionsY()[i],
                           sizeof(float));
                file.write((const char*)&_eventSource->getPositionsZ()[i],
                           sizeof(float));
            }
            if (file.good())
                LBINFO << "Point Sprite positions written as " << pspFile
                       << std::endl;
        }
    }

    void _writePointSpriteIntensities(const std::string& filename) const
    {
        const std::string& psiFile(filename + ".psi");
        std::ofstream file(psiFile.c_str(), std::ios::binary);
        if (file.is_open())
        {
            for (size_t i = 0; i < _eventSource->getNumEvents(); ++i)
                file.write((const char*)&_eventSource->getValues()[i],
                           sizeof(float));
            if (file.good())
                LBINFO << "Point Sprite intensities written as " << psiFile
                       << std::endl;
        }
    }

    std::string _outputFile;
    ::fivox::EventSourcePtr _eventSource;
};

int main(int argc, char* argv[])
{
    ComputeVSD app;
    if (!app.parse(argc, argv))
        return EXIT_SUCCESS;

    app.sample();
}
