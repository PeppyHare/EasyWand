cmake_minimum_required (VERSION 2.8)
# We name this project "EasyWand" as a successor to the MATLAB-implemented EasyWand4.  
project (EASYWAND)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Using Eigen for linear algebra
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# add the executable
add_executable(exec easyWand.cpp)

target_link_libraries(exec ${VTK_LIBRARIES})

enable_testing()

add_test (StillRuns exec RGBsetup_profile.txt wandpts.csv bkgdPts.csv)