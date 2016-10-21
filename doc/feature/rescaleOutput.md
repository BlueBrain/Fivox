Rescaling output voxel values {#rescaleOutput}
=============================

Since Fivox makes all the computations in floating point values, and supports
different output types in the voxelize tool (just uint8 in the Livre
DataSource), we need a way to correctly rescale the resulting values, based
on the input data range and the output type range.

## Requirements

* The internal computations should always be done in floating point values.
* The data source for Livre must always convert the resulting volume from
  float to uint8_t values.
* The voxelize tool will always write a volume in the specified output data
  type, converting it from float if needed.
* The process of rescaling the volume data should always consider the output
  min and max values (known for each data type, e.g. 0-255 for uint8_t), and
  the data range derived from the input (min and max values present after the
  voxelization process in floating point values).
* The conversion should always be done at the end of the process, after all the
  events' contribution has been considered to compute each voxel's value.
* The min and max values of the data range derived from the input
  should be defined from the beginning, based on biological values and/or actual
  data. Since this is very hard to know in advance, specially for time-dependent
  data, a way to manually define this values should be provided. In any case,
  there should be default values.

## Implementation

Floating point values are already used along the whole process. And the
voxelize tool is already taking care of the rescaling when the output data type
is different than float.

In the EventFunctor (base class for the functors) there is a _scale method,
used on each voxel to scale and clamp the value to the output data range. There
are several problems with the current implementation (see fieldFunctor, for
example):

* _scale assumes that the value is always normalized between 0 and 1, which
  might not be true, depending on the value of the 'magnitude' parameter.
* In the case of voltages, it's always shifting the event values by
  brion::MINIMUM_VOLTAGE, to make all the values positive. This is wrong, as
  soon as we start adding voltages together for the computation of one specific
  voxel, we need to know their sign.

The new implementation should get rid of these shift and scale operations in the
functors, and use always the raw values of the report (or the result of the
corresponding operation for an specific use case, e.g. LFP). All the conversion
operations should be done only at the end of the process, once we make sure that
the conversion won't affect the result of the computation.

The 'magnitude' parameter in the input URI is not enough for a correct rescaling
operation. It was added with the intention to hold the value for the input data
range, but this assumes that the starting value is 0, incorrect as soon as we
start dealing with summations of voltages with different signs, for example.

So two new parameters should be added, 'inputMin' and 'inputMax'. And
potentially remove the 'magnitude'. Example:

    voxelize --volume fivoxlfp:///path/to/BlueConfig?report=currentReport&target=targetName&inputMin=-50&inputMax=100

The operation to be performed in order to rescale a value from the input data
range to an output data type range is as follows:

    outputValue = (inputValue - inputMin) * (outputMax - outputMin) / (inputMax - inputMin) + outputMin

## Issues

### 1: Rescale on the full volume or in the functor (on each voxel)?

Resolved: Rescale the full volume at once, after it's been generated.

We can do this operation at the end of the functor, once the summation has been
done for a specific voxel, but we can also do it at the very end, once the
full volume has been generated. ITK provides a filter that does this for us
(IntensityWindowingImageFilter), so we can make use of it easily, adding
this rescale filter as an additional step between the generation of the float
volume and its final use (writing in the case of the voxelize tool, or passing
it to Livre in the case of the data source).

Update: Rescale in the functor (on each voxel).

After a closer review, we noticed the performance implications of using a
FloatVolume in the Livre data source. The only acceptable solution in that case
is to use a ByteVolume from the beginning, so the scaling needs to be done
in the functors. In that case, the ITK filter mentioned above would only be used
in the voxelize app, and the new solution would implement the same behavior
inside the EventFunctor class, deriving all the functors from it.
