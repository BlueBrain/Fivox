/**
 * Copyright (c) BBP/EPFL 2005-2014
 *                        Stefan.Eilemann@epfl.ch
 * All rights reserved. Do not distribute without further notice.
 */

#include <hello/world.h>
#include <hello/version.h>

#include <iostream>

namespace hello
{
void World::greet()
{
    std::cout << "Hello world version " << Version::getRevString()
              << std::endl;
}

int World::getN( const int n )
{
    /// \todo Try harder
    /// \bug Only works for integers
    return n;
}

}
