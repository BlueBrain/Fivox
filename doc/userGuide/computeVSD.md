VSD computation {#computeVSD}
===============

The compute-vsd application was developed with the goal of helping scientists
to do their experiments and research in the context of in-silico VSD imaging.
The design and validation process was done in collaboration with them.

It uses the existing Fivox library, with a few modifications and additions, such
as a new loader (VSDLoader).

## Usage

    compute-vsd --volume 'fivoxvsd://BlueConfig?report=voltages&target=Column&areas=/path/to/report/area.bbp&dt=0.1'
                --curve /path/to/dyecurve.txt --depth 2000 [--interpolate-attenuation]
                --v0 -65 --g0 10000 [--ap-threshold -55] [--export-volume] [--export-point-sprites]
                --sigma 0.001 --sensor-res 512 --sensor-dim 1000
                --frame[s] '0 10' (or alternatively, --time[s] '0 100') --output /path/to/output/prefix_

### Parameters:

##### Volume (fivoxvsd://)

* BlueConfig: Path to the BlueConfig file
* report: Name of the voltage report in the BlueConfig
* target: Name of the Cell Target
* areas: Path to the area report file
* dt: Timestep between requested frames in milliseconds
* gidFraction: Take random cells from a fraction [0,1] of the given target

##### VSD computation

* --curve: Path to the dye curve file. If not specified, the VSD values are not
attenuated
* --depth: Depth of the attenuation curve area of influence, in micrometers. It
also defines the Y-coordinate at which it starts being applied, down until y=0
* --interpolate-attenuation: If specified, linearly interpolate the attenuation
values from the dye curve file. Use nearest-neighbor otherwise (default).
* --v0: Resting potential in millivolts
* --g0: Multiplier for surface area in background fluorescence term
* --ap-threshold: Action potential threshold in millivolts
* --export-volume: If specified, generate a 3D volume containing the VSD signal
(.mhd and .raw files)
* --export-point-sprites: If specified, generate a set of files describing the
VSD events as point sprites (.psh, .psp and .psi files)

##### Beer-Lambert projection

* --sigma: Absorption + scattering coefficient (units per micrometer) in the
Beer-Lambert law. It must be a positive value
* --soma-pixels: Produce a text file with the GID, 3D position and corresponding
pixel coordinates in the final 2D image for all the cells loaded

##### Common

* --sensor-res: Number of pixels per side of the square sensor
* --sensor-dim: Length of side of the square sensor in micrometers
* --frame[s]: Frame[s] to load in the report
* --time[s]: Timestamp[s] (milliseconds) to load in the report
* --output: Path to the directory which will store both the 2D projection data
and optionally the 3D volume containing the VSD values

## Algorithm

The basic idea is to generate a set of events in 3D space, each of them
corresponding to a compartment in the neuron morphology, and then update these
events with the computed VSD value, based on the reported voltage from the
simulation.

The 3D space will then be sampled into a homogeneous volume, resulting in a set
of cubic voxels, each of them containing the aggregated VSD value of all the
events falling within the voxel extent.

Finally, the contents of the 3D volume will be projected vertically onto a
2D surface that represents the pial surface of the brain.

All the steps in the process are detailed next.


### Creation of the events

The first step is to read the circuit information from the BlueConfig file,
using Brion, and load the morphologies corresponding to the target specified. At
the same time, we read the voltage report, and obtain the number of compartments
per section, for all the sections in each of the morphologies. With this
information we can compute the position of the center of the compartments,
by linearly interpolating the sections coordinates.

Each of these compartments becomes an event that will eventually be updated and
evaluated, with an initial value of 0.


### Calculation of the raw VSD signal from simulation data

Then, and depending on the number of frames/times specified as a command line
argument, the corresponding timestamp is loaded in the voltage report. If the
number of compartments that is loaded from the simulation differs from the
number of areas in the area report, then the application will throw an
exception, as this is a requirement for the correct behaviour of the algorithm.

At this point, all the events will be updated with the VSD value as follows:

> VSD = (V - V0 + G0) * SA * AttenuationFactor

being __V__ the corresponding voltage from the simulation report; __V0__ the
specified resting potential; __G0__ the specified area multiplier; and __SA__
the compartment's surface area.

The value of V can be modified, if spike filtering is enabled. For experimental
purposes, it is possible to set an artificial threshold for the spiking
activity: if the input voltage goes above the specified threshold value, the
signal is cropped.

> V = min(V, actionPotentialThreshold)

The attenuation factor is a variable that depends on several factors. If a dye
curve file is not specified as input, it will always be 1, so no attenuation is
applied. It also depends on the Y-coordinate of each of the compartment and the
depth of the loaded circuit (modifiable by an input parameter). Its value is
computed as follows:

1. The values in the input dye curve file are normalized so they go between 0
and 1.
2. If the compartment is above the specified depth value, the first attenuation
factor in the input dye curve file is applied.
3. If the compartment is below 0, the last attenuation factor in the input dye
curve file is applied.
4. If the compartment is within the specified depth, the circuit space is
vertically subdivided in as many regions as values are in the file. Each of the
attenuation factors in the file will be applied to the compartments that fall in
the corresponding region.
5. When the interpolate-attenuation option is specified, the attenuation factor
applied to the VSD value s the result of lineraly interpolating the two closest
values in the input dye curve file, so the curve is smoother.

If the _export-point-sprites_ command line option is specified, the resulting
information (VSD events) will be written to disk in the form of point sprite
files.


### Voxelization

Based on two input parameters, _sensor-res_ and _sensor-dim_, we compute the
desired resolution for the resulting 3D volume, setting the size of the voxels.
In the X and Z axes, the extent of the volume corresponds to the value specified
in sensor-dim, in micrometers; and the number of voxels per side is determined
by sensor-res. The origin of the resulting volume corresponds to the center of
the bounding box of the soma positions.

All the events are evaluated, computing the corresponding voxel indices based on
the event positions, and summing their values together when more than one share
a voxel index. The original floating point value is kept, being 0 the default
for empty voxels.

At this point, if the _export-volume_ command line option is specified, the
resulting 3D volume is written to disk.


### Projection of VSD data to the surface of brain

To generate the projection of the data from the 3D volume onto a 2D image
(representing the plane that rests on the pial surface of the brain), the
Beer-Lambert law is used. To do that, we accumulate all the voxel values along
the Y direction, resulting in a 2-dimensional squared array of floating point
values, with _sensor-res_ pixels per side.

For that, an ITK image filter was implemented. This filter does, for each of the
pixels in the final VSD projection:

> 2DValue = sum(3Dvalue_j * exp(-sigma * depth_j))

being __j__ the index of each of the voxels in the Y axis; __3Dvalue__ the
content of a specific voxel; __sigma__ a coefficient specified by the user;
and __depth__ the depth of the voxel in micrometers, starting from 0 at the top
of the volume.

If the _soma-pixels_ command line option is specified, a text file will be
written to disk, containing the GID, 3D position and corresponding pixel
coordinates in the final 2D image for the somas of all the cells loaded.

## File formats

#### Dye curve file:

ASCII file containing all the attenuation factors, one value per line, from the
top to the bottom, being the first attenuation value the one that will be
applied to the top coordinates of the data.

Example, attenuating more in the middle part:

```cpp
1.00
0.96
0.82
0.70
0.58
0.45
0.31
0.20
0.54
0.83
0.95
```

#### VSD point sprite files:

A metadata header ASCII file (.psh), a binary file containing the positions for
all the VSD events (.psp), and one binary file containing all the event values
for each generated frame (.psi).

#### Output 3D volume:

A metadata header file (.mhd) and the binary file containing the actual volume
values (.raw), that can be opened from any volume rendering application, i.e.
Livre or Paraview.

#### Output 2D image projection:

A VTK file containing a metadata header section and a binary section with the
floating point values of the 2D projection. Readable as a NumPy array from
Python, as follows:

```py
import vtk
from vtk.util import numpy_support
import numpy

reader = vtk.vtkStructuredPointsReader()
reader.SetFileName("image.vtk")
reader.ReadAllScalarsOn()
reader.Update()

points = reader.GetOutput().GetPointData().GetScalars()
array = numpy_support.vtk_to_numpy(points)
```

#### Soma positions and pixel coordinates:

ASCII file containing all GIDs, 3D positions and corresponding pixel coordinates
in the final 2D projection, for all the somas in the scene, with one cell per
line.

It also includes a header section with a brief explanation of the format, file
version, and the library used to generate it. Example with 10 cells:

```cpp
# Soma position and corresponding pixel index for each cell, in the following format:
#     gid [ posX posY posZ ]: i j
# File version: 1
# Fivox version: 0.6.0
10 [     50.115       1971     61.384 ]: 258 256
20 [     54.818     1997.1     71.462 ]: 261 262
30 [     50.896     1760.8     56.659 ]: 258 254
40 [     38.099     1807.9     30.819 ]: 252 241
50 [     19.888     1755.2     69.948 ]: 243 261
60 [     42.529     1730.5     50.998 ]: 254 251
70 [     59.509     1742.3     50.691 ]: 263 251
80 [     43.298     1826.1     57.443 ]: 255 254
160 [     52.034     1595.8     55.536 ]: 259 253
170 [     29.455     1451.9     28.081 ]: 248 239
```
