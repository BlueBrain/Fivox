User Guide {#Userguide}
============

Fivox is a volume generation tool but it can be coupled with [Livre]
(https://bbp.epfl.ch/documentation/code/Livre-0.2/index.html) for real time
rendering.

# Installation {#Installation}

Clone Fivox:
~~~
git clone https://bbpcode.epfl.ch/code/viz/Fivox
~~~

Create your build folder:
~~~
cd Fivox
~~~
~~~
mkdir Build
~~~

For debug build run cmake like that:
~~~
cd Build
~~~
~~~
cmake ..
~~~

For release build run cmake like that:
~~~
cd Build
~~~
~~~
cmake -DCMAKE_BUILD_TYPE=Release ..
~~~

Build fivox with "n" being the number of thread you want to use:
~~~
make -jn
~~~

# Usage {#Usage}

@snippet fivox/livre/dataSource.h Usage
