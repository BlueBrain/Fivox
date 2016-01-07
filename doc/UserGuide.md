User Guide {#Userguide}
============

Fivox is a volume generation tool but it can be coupled with [Livre]
(https://github.com/BlueBrain/Livre#readme) for real time
rendering.

# Installation {#Installation}

Build Fivox from source:
~~~
git clone https://bbpcode.epfl.ch/code/viz/Fivox
mkdir Fivox/build
cd Fivox/build
cmake ..
make
~~~

# Usage {#Usage}

The voxelize command line tool supports the following parameters:

@snippet apps/voxelize.cpp Parameters
