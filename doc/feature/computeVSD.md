VSD computation - User guide
============================

The compute-vsd application was developed with the goal of helping scientists
to do their experiments and research in the context of in-silico VSD imaging.
The design and validation process was done in collaboration with them.

It uses the existing Fivox library, with a few modifications and additions, such
as a new loader (VSDLoader).

## Usage

    compute-vsd --volume 'fivoxvsd://BlueConfig?report=voltages&target=Column&areas=/path/to/report/area.bbp&dt=0.1'
                --curve /path/to/dyecurve.txt --depth 2000 [--interpolate-attenuation]
                --v0 -65 --g0 10000 [--ap-threshold -55] [--export-volume]
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
* --interpolate-attenuation: If specified, interpolate the attenuation values
from the dye curve file
* --v0: Resting potential in millivolts
* --g0: Multiplier for surface area in background fluorescence term
* --ap-threshold: Action potential threshold in millivolts
* --export-volume: If specified, generate a 3D volume containing the VSD signal
(.mhd and .raw files)

##### Beer-Lambert projection

* --sigma: Absorption + scattering coefficient (units per micrometer) in the
Beer-Lambert law. It must be a positive value

##### Common

* --sensor-res: Number of pixels per side of the square sensor
* --sensor-dim: Length of side of the square sensor in micrometers
* --frame[s]: Frame[s] to load in the report
* --time[s]: Timestamp[s] (milliseconds) to load in the report
* --output: Path to the directory which will store both the 2D projection data
and optionally the 3D volume containing the VSD values

## Algorithm

### Creation of the events

The first step is to read the circuit information from the BlueConfig file,
using Brion, and load the morphologies corresponding to the target specified. At
the same time, we read the voltage report, and obtain the number of compartments
per section, for all the sections in each of the morphologies. With this
information we can compute the position of all the compartments in the report,
by linearly interpolating the sections coordinates.

Each of these compartment become an event that will eventually be updated and
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


### Voxelization

Based on two input parameters, _sensor-res_ and _sensor-dim_, we compute the
desired resolution for the resulting 3D volume, setting the size of the voxels.
In the X and Z axes, the extent of the volume corresponds to the value specified
in sensor-dim, in micrometers; and the number of voxels per side is determined
by sensor-res. The origin of the resulting volume corresponds to the center of
the bounding box of the soma positions.

All the events are evaluated, computing the corresponding voxel indices based on
the event center positions, and summing their values together when more than one
share a voxel index. The original floating point value is kept, being 0 the
default for empty voxels.

At this point, if the _export-volume_ command line option is specified, the
resulting 3D volume is written to disk.


### Projection of VSD data to the surface of brain

To generate the projection of the data from the 3D volume into a 2D image that
represents the surface of the brain, the Beer-Lambert law is used. To do that,
we accumulate all the voxel values along the Y direction, resulting in a
2-dimensional squared array of floating point values, with _sensor-res_ pixels
per side.

For that, an ITK image filter was implemented. This filter does, for each of the
pixels in the final VSD projection:

> 2DValue = sum(3Dvalue_j * exp(-sigma * depth_j))

being __j__ the index of each of the voxels in the Y axis; __3Dvalue__ the
content of a specific voxel; __sigma__ a coefficient specified by the user;
and __depth__ the depth of the voxel in micrometers, starting from 0 at the top
of the volume.


## File formats

#### Dye curve file:

ASCII file containing all the attenuation factors, one value per line, from the
top to the bottom, being the first attenuation value the one that will be
applied to the top coordinates of the data.

Example, attenuating more in the middle part:

```
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

#### Output 3D volume:

A metadata header file (.mhd) and the binary file containing the actual volume
values (.raw), that can be opened from any volume rendering application, i.e.
Livre or Paraview.

#### Output 2D image projection:

A VTK file containing a metadata header section and a binary section with the
floating point values of the 2D projection. Readable as a NumPy array from
Python, as follows:

```
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
