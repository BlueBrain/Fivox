/* Copyright (c) 2014-2016, EPFL/Blue Brain Project
 *                          Jafet.VillafrancaDiaz@epfl.ch
 *                          Juan.Hernando@epfl.ch
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

#ifndef FIVOX_HELPERS_H
#define FIVOX_HELPERS_H

#include <fivox/eventSource.h>

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>
#include <brion/compartmentReport.h>
#include <brion/types.h>

#include <lunchbox/log.h>

namespace fivox
{
namespace helpers
{
/** Tuple of buffer offset, cell index, section ID, compartment counts */
typedef std::tuple<size_t, uint32_t, uint32_t, uint16_t> MappingElement;
typedef std::vector<MappingElement> FlatInverseMapping;

/**
 * Computes a report mapping that enumerates compartments in the same
 * order they appear in the data buffer.
 * @param report The report from which the mapping is computed.
 * @return A list of tuples (offset, cell index, section ID, compartment count).
 */
inline FlatInverseMapping computeInverseMapping(
    const brion::CompartmentReport& report)
{
    FlatInverseMapping mapping;
    mapping.reserve(report.getBufferSize());

    const auto& offsets = report.getOffsets();
    const auto& counts = report.getCompartmentCounts();
    for (size_t i = 0; i != offsets.size(); ++i)
    {
        for (size_t j = 0; j != offsets[i].size(); ++j)
        {
            const size_t count = counts[i][j];
            if (count != 0)
                mapping.push_back(std::make_tuple(offsets[i][j], i, j, count));
        }
    }
    std::sort(mapping.begin(), mapping.end());

#ifdef DEBUG_INVERSE_MAPPING
    size_t offset = 0;
    for (auto i : mapping)
    {
        assert(offset == std::get<0>(i));
        offset += std::get<3>(i);
    }
#endif
    return mapping;
}

/**
 * Add one event per simulation compartment to the given event source.
 * The compartment counts are obtained from the report mapping. The event
 * positions are computed from the morphology list.
 *
 * @param morphologies The list of morphologies. The morphology present at each
 *        index must correspond to the cell at the same index in the report
 *        mapping.
 * @param report The report from which the compartments per section are obtained
 * @param output The output event source. Events are added in the morphology
 *        iteration order, starting with the soma and then all the dendrites.
 * @param somasOnly Specify whether the events will be created for the somas
 *        only or for all the compartments. False by default (load all).
 */
inline void addCompartmentEvents(
    const brain::neuron::Morphologies& morphologies,
    const brion::CompartmentReport& report, EventSource& output,
    const bool somasOnly = false)
{
    size_t size = 0;
    const auto& mapping = computeInverseMapping(report);
    // first loop over the mapping to compute the number of elements
    for (const auto& i : mapping)
    {
        size_t offset;
        uint32_t cellIndex;
        uint32_t sectionId;
        uint16_t compartments;
        std::tie(offset, cellIndex, sectionId, compartments) = i;

        if (somasOnly && sectionId != 0)
            continue;

        size += compartments;
    }
    output.resize(size);

    size_t index = 0;
    // second loop to add the actual events
    for (const auto& i : mapping)
    {
        size_t offset;
        uint32_t cellIndex;
        uint32_t sectionId;
        uint16_t compartments;
        std::tie(offset, cellIndex, sectionId, compartments) = i;

        const auto& morphology = *morphologies[cellIndex];

        if (sectionId == 0)
        {
            const auto& soma = morphology.getSoma();
            for (size_t k = 0; k != compartments; ++k)
                output.update(index++, soma.getCentroid(),
                              soma.getMeanRadius());
            continue;
        }

        if (somasOnly)
            continue;

        brion::floats samples;
        samples.reserve(compartments);
        // normalized compartment length
        const float normLength = 1.f / float(compartments);
        for (float k = normLength * .5f; k < 1.0; k += normLength)
            samples.push_back(k);

        const auto& neuronSection = morphology.getSection(sectionId);

        // actual compartment length
        const float compartmentLength = normLength * neuronSection.getLength();

        const auto& points = neuronSection.getSamples(samples);
        for (const auto& point : points)
        {
            output.update(index++, point.get_sub_vector<3, 0>(),
                          compartmentLength * .2f);
        }
    }
}
}
}
#endif
