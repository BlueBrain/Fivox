# Copyright (c) 2012 Daniel Nachbaur <daniel.nachbaur@epfl.ch>

# Info: http://www.itk.org/Wiki/CMake:Component_Install_With_CPack

set(CPACK_PACKAGE_VENDOR "bluebrain.epfl.ch")
set(CPACK_PACKAGE_VERSION ${VERSION})
set(CPACK_PACKAGE_NAME_EXTRA "bbp")

include(oss/CommonCPack)
include(BBPPackages)
