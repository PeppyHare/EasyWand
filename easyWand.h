#ifndef _EASYWAND_H
#define _EASYWAND_H

// #include <vector>                       // Wand/bkgd pts stored as vectors
// #include "csv_istream_iterator.h"       // Use iterator to read .csv files

class easyWand{
public:
    // Default constructor
    easyWand(const std::string camProfile = "", const std::string wandPtsFile = "", const std::string bkgdPtsFile = "");

    // Destructor
    // Probably make this later.....  >.> using default for now


    const std::vector<std::vector<double> > getWandPoints() const;
    // Getter method for wand point data

    const int getNumCams() const;
    // Getter method for number of cameras

    void readCameraProfiles(const std::string camProfile);
    // Read in the camera profiles from a .txt file.  File format is 
    // Camera #  |  X res  |  Y res  | X pp  |  Y pp  |  Primary?  
    // One row per camera 

    void readWandPoints(const std::string wandPtsFile, 
        std::vector<std::vector<double> > &wandPts);
    // Read wand points in from csv file using fstream and convert to double using sstream 
    // Wand points are read in as a 1d array, so need to mod out by the number of cameras

    void plotWandPoints(const std::vector< std::vector<double> > wandPts, const int camera);
    // Create a VTK renderer view for plotting the wand points of a specific camera
    // Should default to the first camera
    // TODO: Enable switching view to other cameras via GUI

    void computeCalibration();
    // Do the actual calibration routines.  May need to break this into smaller chunks later for readability



private:
    int numCameras;
    // The number of cameras to calibrate (int)
    
    double wandLen;
    // The calibration wand length (units are the universal calibration units -- If wand length specified in meters, then all distances are returned in meters)

    double wandScore;
    // RMS error in the wand point reconstruction.  Lower error is better, preferable < 1-2
    
    std::vector<double> ppts;
    // Principal point vector.  [2nCams x 1] vector containing (X ppt, Y ppt) 
    // for each

    std::vector<int> resolutions;
    // Resolution vector.  [2nCams x 1] vector containing ( X res, Y res ) 
    // for each camera
    
    std::vector<std::vector<double> > wandPts;
    // The wand points.  Formatted as
    // |cam1/pt1x | cam1/pt1y | cam2/pt1x | cam2/pt1y | cam1/pt2x | cam1/pt2y | cam2/pt2x | cam2/pt2y|

    std::vector<std::vector<double> > bkgdPts;
    // The background points.  Formatted as 
    // |cam1/pt1x | cam1/pt1y | cam2/pt1x | cam2/pt1y | cam1/pt2x | cam1/pt2y | cam2/pt2x | cam2/pt2y|

    std::vector<double> foc;
    // Focal lengths of the cameras.  [nCams x 1] vector containing the focal lengths in pixels

    std::vector<std::vector<double> > w1;
    // 3D reconstruction of side 1 of the wand

    std::vector<std::vector<double> > w2;
    // 3D reconstruction of side 2 of the wand 
};

// Utility functions used outside of the easyWand class (for math and Eigen)


Eigen::MatrixXd VectToEigenMat(const std::vector<double> inputvector);
// Straightforward copy constructor from std::vector<double> to Eigen::VectorXd

Eigen::MatrixXd VectToEigenMat(const std::vector< std::vector<double> > inputvector);
// Straightforward copy constructor from std::vector<double> to Eigen::MatrixXd 

Eigen::MatrixXd computeF(const Eigen::MatrixXd &ptNorm);
// Computes the fundamental matrix F via singular value decomposition with normalization as reccommended in Hartley-Zisserman

double twoCamCal(const Eigen::MatrixXd ptNorm, Eigen::Matrix3d camMatrix, Eigen::Matrix3d rMat, Eigen::Vector3d tVec);
// Computes the rotation matrix and translation vector between two cameras
// from a shared set of [u,v] coordinates.
//
// Inputs:
//  ptNorm - [n,4] array of the form [c1U1,c1V1,c2U1,c2V2]
//                                   [ ................. ]
//                                   [c1Un,c1Vn,c2Un,c2Vn]
//
//  camMatrix - [3,3] rank 2 assumed camera matrix
//
// Outputs:
//  rMat - [3,3] rotation matrix
//  tVec - [3] translation vector
// 
//  Return:
//  c - Quality score; lower is better
//
// Note that the rotation and translation is for camera1 with respect to
// camera2, i.e. camera2 sits at R=[0,0,0;0,0,0;0,0,0] and T=[0,0,0].


Eigen::MatrixXd triangulate(const Eigen::Matrix3d rotM, const Eigen::Vector3d tv, const Eigen::MatrixXd ptNorm);
// Try and best triangulate a set of 3D points from a camera rotation matrix (rotM), a translation vector (tv), and a set of 2D shared points (ptNorm).  Of course this is done with another singular value decomposition, to try and minimize residuals



#endif