
set(FIVOX_PACKAGE_VERSION 0.1)
set(FIVOX_REPO_URL ssh://bbpcode.epfl.ch/viz/Fivox)
set(FIVOX_DEPENDS BBPSDK BBPTestData REQUIRED ITK Boost vmmlib)
#see ubuntu bug 1328924: set(FIVOX_DEB_DEPENDS libinsighttoolkit4-dev)
set(FIVOX_BOOST_COMPONENTS "unit_test_framework program_options system")

if(CI_BUILD_COMMIT)
  set(FIVOX_REPO_TAG ${CI_BUILD_COMMIT})
else()
  set(FIVOX_REPO_TAG master)
endif()
set(FIVOX_FORCE_BUILD ON)
set(FIVOX_SOURCE ${CMAKE_SOURCE_DIR})