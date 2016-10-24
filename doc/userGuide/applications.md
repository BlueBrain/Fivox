Applications
============

High-level description of the main applications currently supported by Fivox.

## Validated applications

Real scientific use cases, coming from user requests and finished after a
collaboration with scientists:

#### Local Field Potential

Compute the LFP signal at the extracellular space of a given circuit, based on
a simulation current report
([more info](https://en.wikipedia.org/wiki/Local_field_potential)).

The output is generated as a 3D volume, containing each of the voxels the LFP
value corresponding to the position at the center of the voxel.

For convenience, a separate tool (sample-point) is provided, to generate the
time series of the output voltage at a specific 3D point. In this case the space
is not voxelized, and the computation is done for a single point, so the
resulting values are more accurate, as an exact 3D position is used as input.
The output is a text file with one value per timestep, being possible to plot
this values in order to visualize the evolution of the voltage over time as a
2D graph.


#### Voltage-Sensitive Dye

Generate the in-silico VSD optical imaging of a given circuit, based on the
compartment areas and simulation voltage report
([more info](https://en.wikipedia.org/wiki/Voltage-sensitive_dye)).

A separate tool is provided, with the ability to generate a 2D image with the
floating point values, representing the projection on the pial surface of the
brain; and also a 3D volume containing the VSD values for each of the
compartments (before the projection step).


#### Synapse densities

Generate a 3D volume with the synapse densities per voxel, for the given pre
and post-synaptic targets.


## Potential applications

Solutions that could be (or already are) implemented, but without a clear
application, as they don't fulfill any real scientific need:

#### Spiking frequencies

Generate a 3D volume with the spike frequencies per voxel, for the given
circuit and spike report.


#### Voltage fields

Generate a 3D volume representing the influence of the compartment voltages in
the extracellular space (e.g. applying a square falloff) for a given circuit.
