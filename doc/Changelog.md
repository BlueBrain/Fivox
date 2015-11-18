Changelog {#changelog}
=========

# git master {#master}

* [19230](https://bbpcode.epfl.ch/code/#/c/19230/)
  Add support for volume time series generation for voxelize tool.
* [19188](https://bbpcode.epfl.ch/code/#/c/19181/)
  Bug fixes in compartment report indexing in compartment and VSD loaders.
* [19077](https://bbpcode.epfl.ch/code/#/c/19077/)
  Dt defaults to experiment/report dt, duration to 10.

# Release 0.3 (9-Nov-2015){#Release03}

* [18462](https://bbpcode.epfl.ch/code/#/c/18462/)
  Added new frequency functor, make more parameters configurable in URI
* [15814](https://bbpcode.epfl.ch/code/#/c/15814/)
  Fix underflow events by using the newly introduced MINIMUM_VOLTAGE in the
  voltage-related loaders
* [15670](https://bbpcode.epfl.ch/code/#/c/15670/)
  Tweak livre rendering parameters: blocks are now max 64 MB, and their size
  in voxels is always multiple of 8 for better alignment
* [14496](https://bbpcode.epfl.ch/code/#/c/14496/)
  New synapse loader to visualize synapse densities
* [14204](https://bbpcode.epfl.ch/code/#/c/14204/)
  Add generic voxelize tool and URIHandler
* [14204](https://bbpcode.epfl.ch/code/#/c/14204/)
  Fix bug in loaders: the event source loaded always a specific frame,
  even when a time stamp was specified.
* [14181](https://bbpcode.epfl.ch/code/#/c/14181/)
  New dye curve parameter for VSD reports in the Livre data source
* [14203](https://bbpcode.epfl.ch/code/#/c/14203/)
  Add missing method in SpikeLoader to load data for a specific timestamp
* [14181](https://bbpcode.epfl.ch/code/#/c/14181/)
  Fix computation of event sampling values for squared distances below 1
* [16290](https://bbpcode.epfl.ch/code/16290/)
  The targets can be given as a part of URL queries ( i.e.: target=Column )


# Release 0.2 (9-Jul-2015){#Release02}

* [13240](https://bbpcode.epfl.ch/code/#/c/13240/)
  [5296](https://bbpcode.epfl.ch/code/#/c/5296/)
  Add time and animation support
* [13383](https://bbpcode.epfl.ch/code/#/c/13383/)
  Fixed LOD bug, optimized tree size acording to the data bounding box
* [13762](https://bbpcode.epfl.ch/code/#/c/13762/)
  5x speedup for spike sampling
* [13877](https://bbpcode.epfl.ch/code/#/c/13877/)
  Fix voxelizeVSD tool to load the specified time
* [13669](https://bbpcode.epfl.ch/code/#/c/13669/)
  Use non-negative voltage in vsd loader
* [13102](https://bbpcode.epfl.ch/code/#/c/13102/)
  Allow different targets for spikes, make report for compartment & soma
  accessible from URI
* [10636](https://bbpcode.epfl.ch/code/#/c/10636/)
  Adding depth attenuation property to vsd
* [13947](https://bbpcode.epfl.ch/code/#/c/13947/)
  Use the circuit target when no target is provided
* [12973](https://bbpcode.epfl.ch/code/#/c/12973/)
  Change datasource schemes to lowercase
* Documentation updated

# Release 0.1 (30-Apr-2015){#Release01}

* Initial version
