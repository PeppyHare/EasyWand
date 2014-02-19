###### EASYWAND C++ VERSION!
===============================================================================
    This is a C++ implementation of a MATLAB project I previously contributed to.  EasyWand is meant to be a user-friendly camera calibration toolbox, largely geared towards academic use.  As such, the ultimate aim is for an easy setup, cross-platform functionality, thorough docs, efficient runtime, and lots of tools.  For now, this is mainly a learning experience for me, as a getting-to-know you experience with C++.

### Necessary Installs:
    At this point, it's necessary to have installed [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) and [VTK](http://www.vtk.org/).  Soon, these will be bundled along with the install, and everything will be nicer.  Until I'm much better with CMake, you're on your own.

### Quickstart Guide:
## -CMake:
The CMakeLists.txt file should be all you need.  Just run 
```cmake /path/to/EasyWand/
make all'''

## -GCC
If you want to link to your own VTK and Eigen3 libraries, be my guest.

More easy installs coming later...


There is some sample data sitting right in the top-level directory.  There is a sample camera calibration profile file (RGBsetup_profile.txt), a sample set of digitized calibration points (wandpts.csv), and a set of background points (bkgdpts.csv).  Usage will probably change violently over the next few weeks, so just take a peek at the source. 

_Evan Bluhm_
_bluhme3@gmail,com_