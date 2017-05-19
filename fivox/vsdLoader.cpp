/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan Hernando <jhernando@fi.upm.es>
 *                          Daniel.Nachbaur@epfl.ch
 *
 * This file is part of Fivox <https://github.com/BlueBrain/Fivox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "vsdLoader.h"

#include "helpers.h"
#include "uriHandler.h"

#include <brain/circuit.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>
#include <brion/brion.h>

#include <cassert>

namespace fivox
{
class VSDLoader::Impl
{
public:
    Impl(EventSource& output, const URIHandler& params)
        : _output(output)
        , _circuit(params.getConfig())
        , _gids(params.getGIDs())
        , _voltageReport(params.getConfig().getReportSource(params.getReport()),
                         brion::MODE_READ, _gids)
        , _areaReport(URI(params.getAreas()), brion::MODE_READ, _gids)
        , _restingPotential(0.f)
        , _areaMultiplier(0.f)
        , _spikeFilter(false)
        , _apThreshold(0.f)
        , _interpolate(false)
    {
        LBINFO << "Loading " << _gids.size() << " morphologies..." << std::endl;
        const auto morphologies =
            _circuit.loadMorphologies(_gids,
                                      brain::Circuit::Coordinates::global);

        LBINFO << "Creating events..." << std::endl;
        helpers::addCompartmentEvents(morphologies, _voltageReport, _output);

        LBINFO << "Loading areas..." << std::endl;
        _areas = _areaReport.loadFrame(0.).get();
    }

    ssize_t load()
    {
        brion::floatsPtr voltages =
            _voltageReport.loadFrame(_output.getCurrentTime()).get();
        if (!voltages)
            return -1;

        if (voltages->size() != _areas->size())
            LBTHROW(
                std::runtime_error("The number of compartments in the "
                                   "voltage report doesn't match the "
                                   "number of areas"));
        for (size_t i = 0; i != voltages->size(); ++i)
        {
            const float voltage = (*voltages)[i];
            _updateEventValue(i, _spikeFilter ? std::min(voltage, _apThreshold)
                                              : voltage,
                              (*_areas)[i]);
        }
        return voltages->size();
    }

    void _updateEventValue(const size_t index, const float voltage,
                           const float area)
    {
        const float positionY = _output.getPositionsY()[index];
        _output[index] = (voltage - _restingPotential + _areaMultiplier) *
                         area * _curve.getAttenuation(positionY, _interpolate);
    }

    EventSource& _output;
    const brain::Circuit _circuit;
    brion::GIDSet _gids;

    brion::CompartmentReport _voltageReport;
    brion::CompartmentReport _areaReport;
    brion::floatsPtr _areas;
    AttenuationCurve _curve;

    AABBf _bboxSomas;        // bounding box of the somas
    float _restingPotential; // resting potential (mV)
    float _areaMultiplier;   // multiplier for surface area in background
                             // fluorescence term
    bool _spikeFilter;  // use the action potential threshold to filter spikes
    float _apThreshold; // action potential threshold (mV)
    bool _interpolate;  // interpolate the attenuation from the dye curve
};

VSDLoader::VSDLoader(const URIHandler& params)
    : EventSource(params)
    , _impl(new VSDLoader::Impl(*this, params))
{
    if (getDt() < 0.f)
        setDt(_impl->_voltageReport.getTimestep());
}

VSDLoader::~VSDLoader()
{
}

void VSDLoader::setCurve(const AttenuationCurve& curve)
{
    _impl->_curve = curve;
}

const brion::GIDSet& VSDLoader::getGIDs() const
{
    return _impl->_gids;
}

const brion::Vector3fs VSDLoader::getSomaPositions() const
{
    return _impl->_circuit.getPositions(_impl->_gids);
}

void VSDLoader::setRestingPotential(const float millivolts)
{
    _impl->_restingPotential = millivolts;
}

void VSDLoader::setAreaMultiplier(const float factor)
{
    _impl->_areaMultiplier = factor;
}

void VSDLoader::setSpikeFilter(const bool enable)
{
    _impl->_spikeFilter = enable;
}

void VSDLoader::setApThreshold(const float apThreshold)
{
    _impl->_apThreshold = apThreshold;
}

void VSDLoader::setInterpolation(const bool interpolate)
{
    _impl->_interpolate = interpolate;
}

Vector2f VSDLoader::_getTimeRange() const
{
    return Vector2f(_impl->_voltageReport.getStartTime(),
                    _impl->_voltageReport.getEndTime());
}

ssize_t VSDLoader::_load(const size_t /*chunkIndex*/,
                         const size_t /*numChunks*/)
{
    return _impl->load();
}
}
