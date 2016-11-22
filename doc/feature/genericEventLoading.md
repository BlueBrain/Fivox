Generic event loading {#genericEventLoading}
=====================

For research and debugging purposes, sometimes we need to quickly visualize
a set of events in 3D space. For example, in a neural circuit with synaptic
connectivity, we are interested in visualizing the position of the post-synaptic
cells; or for validation purposes, we want to check that the LFP computation is
done correctly for a predefined set of events.

Since the variety of use cases is so wide, each of them with its own
particularities, we need to find a generic way of loading any predefined set of
events.

The solution chosen for this feature is the loading of events from file. This
offers users flexibility to generate all types of events in a simple and generic
way, so the tool doesn't need to be adapted to fit the particular needs of any
situation.

A generic event is always defined by its 3D position, radius and value,
regardless of the use case. It could be the cases that the radius and value are
not relevant, but they are always present (e.g. in the synaptic connectivity,
the radius can be 0, and value 1, as we will count each event as one cell).


## Requirements

* Users should be able to specify their own input events, as easily as possible,
from an external source (i.e. file).
* The format of the input file should be clearly defined and documented.
* All the library common capabilities should still be available for their use 
with this feature. For example, generating 3D volumes with different
resolutions and using any available functor.


## API

A TestLoader is already present in Fivox, which generates a set of 7 events in a
fixed position, and is used for basic use cases such as the validation of the
LFP computation in a test case. The input URI in this case is as follows:

    "fivoxtest://?resolution=1&functor=lfp"

What we need for the loading of any number of generic events is very similar to
this, but we need a way to specify the source of our events, instead of
generating them at runtime. We can use the BlueConfig path location as the event
file path. A change in the name might also be convenient, as it is not anymore
just for testing purposes. The loader and schema names are open for discussion.

    "fivox:///absolute/path/to/events/file?resolution=1&functor=lfp"

The path to the input file is optional, behaving like the original TestLoader
and creating the initial 7 events if it is omitted. Please note that, when using
the GenericLoader, the events file path replaces the BlueConfig path (used
in most of the scientific use cases).

For simplicity, and for now, only the GenericLoader can read events from file.
But any EventSource can write this type of files, dumping their contents (all
the events and their corrresponding values that it contains at the moment).
For that, a new method will be implemented in the base class:

    bool EventSource::write( const std::string& filename, bool binary = true );

The corresponding option should also be added in the _voxelize_ application,
so users have access to it.


## File format

For the input file containing the events, two different formats should be
supported, binary for performance, and ASCII for readability, but keeping the
same layout.

#### ASCII

Header containing the version numbers and the format used, followed by all the
events (one event per line).

Example:

```cpp
# Fivox events (3D position, radius and value), in the following format:
#     posX posY posZ radius value
# File version: 1
# Fivox version: 0.6.0
50.115 1971 61.384 1.0 1
54.818 1997.1 71.462 1.0 2
50.896 1760.8 56.659 1.0 3
38.099 1807.9 30.819 1.0 4
19.888 1755.2 69.948 1.0 5
42.529 1730.5 50.998 1.0 6
59.509 1742.3 50.691 1.0 7
43.298 1826.1 57.443 1.0 8
52.034 1595.8 55.536 1.0 9
29.455 1451.9 28.081 1.0 10
```

#### Binary

Two 32-bit values at the beginning (magic and version numbers), followed by
all the events, with five 32-bit floating point values
(posX posY posZ radius value) per event.


## Issues

### 1: What names should we use for the loader and URI schema?

_Resolution: Yes_ GenericLoader and "fivox://".

We can replace the TestLoader by GenericLoader. Other alternatives are:
EventLoader, GenericEventLoader, FileLoader.

As for the URI schema, we have "fivoxtest://", that could be replaced by
simply "fivox://" (which is now used for the CompartmentLoader, but that could
be changed), "fivoxfile://", "fivoxevents://", "fivoxgeneric://".

### 2: Why not extending this feature for any loader, to be able to store the event positions in a cache file after their initial creation?

_Resolution: Yes_ We can write the events from any loader, but read only from
the GenericLoader (for now).

It's an interesting possibility, specially considering the amount of time it
takes in some occasions to just create the events based on a BlueConfig file and
target, for example. The problem is that not every loader works the same way
(for example the SynapseLoader loads by chunks, not everything at once at
construction time), and this addition would also complicate the implementation
and usage of each of the loaders. For the events reading, we should keep this
feature self-contained in the new GenericLoader, at least for now, and then we
see if it is possible to extend it for any use case. For writing, there should
not be any problem in making it available for any loader (the implementation
