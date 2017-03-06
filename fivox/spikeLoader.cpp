/* Copyright (c) 2015-2016, EPFL/Blue Brain Project
 *                          Stefan.Eilemann@epfl.ch
 *                          Jafet.VillafrancaDiaz@epfl.ch
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

#include "spikeLoader.h"
#include "uriHandler.h"

#include <brain/brain.h>
#include <brion/brion.h>

#include <lunchbox/os.h>

using boost::lexical_cast;

namespace fivox
{
class SpikeLoader::Impl
{
public:
    Impl(EventSource& output, const URIHandler& params)
        : _output(output)
        , _spikesStart(0.f)
    {
        const brion::GIDSet& gids = params.getGIDs();

        const brain::Circuit circuit(params.getConfig());
        const brion::Vector3fs& positions = circuit.getPositions(gids);

        size_t i = 0;
        _output.resize(gids.size());
        _spikesPerNeuron.resize(gids.size());
        _gidIndex.resize(*gids.rbegin() + 1);
        for (const uint32_t gid : gids)
        {
            _output.update(i, positions[i], /*radius*/ 0.f);
            _gidIndex[gid] = i++;
        }

        const std::string& spikePath = params.getSpikes();
        _report.reset(new brain::SpikeReportReader(
            spikePath.empty() ? params.getConfig().getSpikeSource()
                              : URI(spikePath),
            gids));
        _spikesEnd = _report->getEndTime();
    }

    void updateTimeRange()
    {
        if (!_report || _report->hasEnded())
            return;

        const auto endtime = _report->getEndTime();
        // This forces the collection of the latest spikes in the stream
        // case, thus updating the end time.
        _report->getSpikes(std::nextafter(endtime,
                                          -std::numeric_limits<float>::max()),
                           endtime);
        // don't update _spikesStart to calculate absolute frame numbers
        // see https://bbpcode.epfl.ch/code/#/c/19337
        _spikesEnd = _report->getEndTime();
    }

    ssize_t load()
    {
        const float start = _output.getCurrentTime();
        lunchbox::setZero(_spikesPerNeuron.data(),
                          _spikesPerNeuron.size() * sizeof(size_t));

        const float end = start + _output.getDuration();
        const size_t numSpikes = _loadSpikes(start, end);

        for (size_t i = 0; i < _spikesPerNeuron.size(); ++i)
            _output[i] = _spikesPerNeuron[i];

        return numSpikes;
    }

    size_t _loadSpikes(const float start, const float end)
    {
        size_t numSpikes = 0;
        for (const auto& spike : _report->getSpikes(start, end))
        {
            ++_spikesPerNeuron[_gidIndex[spike.second]];
            ++numSpikes;
        }

        return numSpikes;
    }

    EventSource& _output;
    float _spikesStart;
    float _spikesEnd;

    // maps GID to its index in the target
    // OPT: no (unordered)map because of constant lookup but 'wastes' memory
    // (container.size() is number of GIDs)
    brion::size_ts _gidIndex;

    // aggregates spikes for each neuron in interval
    // OPT: no (unordered)map because of constant lookup but 'wastes' memory
    // (container.size() is number of GIDs)
    brion::size_ts _spikesPerNeuron;

    std::unique_ptr<brain::SpikeReportReader> _report;
};

SpikeLoader::SpikeLoader(const URIHandler& params)
    : EventSource(params)
    , _impl(new Impl(*this, params))
{
    if (getDt() < 0.f)
        setDt(params.getConfig().getTimestep());
}

SpikeLoader::~SpikeLoader()
{
}

Vector2f SpikeLoader::_getTimeRange() const
{
    _impl->updateTimeRange();
    return Vector2f(_impl->_spikesStart, _impl->_spikesEnd);
}

ssize_t SpikeLoader::_load(const size_t /*chunkIndex*/,
                           const size_t /*numChunks*/)
{
    return _impl->load();
}
}
