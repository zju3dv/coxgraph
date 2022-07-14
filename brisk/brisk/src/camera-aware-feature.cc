/*
 * CameraAwareFeature.cpp
 *
 *  Created on: Jan 1, 2014
 *      Author: lestefan
 */

#include <iostream>  // NOLINT

#include <agast/wrap-opencv.h>
#include <brisk/camera-aware-feature.h>
#include <brisk/brisk-feature.h>
#include <brisk/brisk-descriptor-extractor.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace brisk {

CameraAwareFeature::CameraAwareFeature():_e_C(0,0,0){}

CameraAwareFeature::~CameraAwareFeature() { }

CameraAwareFeature::CameraAwareFeature(
    cv::Ptr<cv::Feature2D> feature2dPtr,
    const/*CAMERA_GEOMETRY_T*/cv::Ptr<cameras::CameraGeometryBase> cameraGeometryPtr,
    double distortionTolerance)
    : _distortionTolerance(distortionTolerance), _e_C(0,0,0) {
  setFeature2d(feature2dPtr);
  setCameraGeometry(cameraGeometryPtr);
}

// setters
void CameraAwareFeature::setFeature2d(
    cv::Ptr<cv::Feature2D> feature2dPtr) {
  _feature2dPtr = feature2dPtr;
}

void CameraAwareFeature::setDistortionTolerance(double distortionTolerance) {
  _distortionTolerance = distortionTolerance;
}

//template<class CAMERA_GEOMETRY_T>
void CameraAwareFeature::setCameraGeometry(
    const/*CAMERA_GEOMETRY_T*/cv::Ptr<cameras::CameraGeometryBase> cameraGeometryPtr) {

  // store the geometry
  _cameraGeometryPtr = cameraGeometryPtr;

  // generate the maps

  // first, project corners to undistorted image, in order to determine its size
  // we assume that the corner rays are farthest apart
  cameras::Point3d p_C_00, p_C_w0, p_C_0h, p_C_wh;
  cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(0, 0), p_C_00);
  cameraGeometryPtr->keypointToEuclidean(
      cameras::Point2d(cameraGeometryPtr->width(), 0), p_C_w0);
  cameraGeometryPtr->keypointToEuclidean(
      cameras::Point2d(0, cameraGeometryPtr->height()), p_C_0h);
  cameraGeometryPtr->keypointToEuclidean(
      cameras::Point2d(cameraGeometryPtr->width(), cameraGeometryPtr->height()),
      p_C_wh);
  p_C_00 = cv::normalize(p_C_00);
  p_C_w0 = cv::normalize(p_C_w0);
  p_C_0h = cv::normalize(p_C_0h);
  p_C_wh = cv::normalize(p_C_wh);

  // original image dimensions
  const size_t width = cameraGeometryPtr->width();
  const size_t height = cameraGeometryPtr->height();

  // figure out focal length
  cameras::Point3d p_C_mc, p_C_pc;
  cameraGeometryPtr->keypointToEuclidean(
      cameras::Point2d(cameraGeometryPtr->width() / 2.0 - 1.0,
                       cameraGeometryPtr->height() / 2.0),
      p_C_mc);
  cameraGeometryPtr->keypointToEuclidean(
      cameras::Point2d(cameraGeometryPtr->width() / 2.0 + 1.0,
                       cameraGeometryPtr->height() / 2.0),
      p_C_pc);
  const double n_focalLength = 1.0
      / ((p_C_pc[0] / p_C_pc[2] - p_C_mc[0] / p_C_mc[2]) / 2.0);

  /*const size_t n_width = (-std::min(p_C_00[0]/p_C_00[2],p_C_0h[0]/p_C_0h[2]) +
   std::max(p_C_w0[0]/p_C_w0[2],p_C_wh[0]/p_C_wh[2]))*n_focalLength +
   0.5;
   const size_t n_height = (-std::min(p_C_00[1]/p_C_00[2],p_C_w0[1]/p_C_w0[2]) +
   std::max(p_C_0h[1]/p_C_0h[2],p_C_wh[1]/p_C_wh[2]))*n_focalLength +
   0.5;

   const double c_x=double(-std::min(p_C_00[0]/p_C_00[2],p_C_0h[0]/p_C_0h[2]))*n_focalLength;
   const double c_y=double(-std::min(p_C_00[1]/p_C_00[2],p_C_w0[1]/p_C_w0[2]))*n_focalLength;*/

  // find out into how many different sub-images to split:
  const double angle_x = std::max(
      acos(
           p_C_00[0] * p_C_w0[0] + p_C_00[1] * p_C_w0[1]
               + p_C_00[2] * p_C_w0[2]),
      acos(
           p_C_0h[0] * p_C_wh[0] + p_C_0h[1] * p_C_wh[1]
               + p_C_0h[2] * p_C_wh[2]));
  const double angle_y = std::max(
      acos(
           p_C_00[0] * p_C_0h[0] + p_C_00[1] * p_C_0h[1]
               + p_C_00[2] * p_C_0h[2]),
      acos(
           p_C_w0[0] * p_C_wh[0] + p_C_w0[1] * p_C_wh[1]
               + p_C_w0[2] * p_C_wh[2]));
  //std::cout<<angle_x/M_PI*180<<std::endl;
  //std::cout<<angle_y/M_PI*180<<std::endl;
  _N_x = angle_x / 2.0 / _distortionTolerance + 1.0;
  _N_y = angle_y / 2.0 / _distortionTolerance + 1.0;
  //std::cout<<_N_x<<","<<_N_y<<std::endl;

  // these are the maps for distortion and undistortion:
  _distort_x_maps.resize(_N_x * _N_y);
  _distort_y_maps.resize(_N_x * _N_y);
  _distort_1_maps.resize(_N_x * _N_y);
  _distort_2_maps.resize(_N_x * _N_y);
  _undistort_x_maps.resize(_N_x * _N_y);
  _undistort_y_maps.resize(_N_x * _N_y);
  _undistort_1_maps.resize(_N_x * _N_y);
  _undistort_2_maps.resize(_N_x * _N_y);
  _undistortedModels.clear();

  // initialize the model lookup
  _cameraModelSelection = cv::Mat::zeros(height, width, CV_8UC1);

  // fill normals first
  std::vector < cv::Vec3d > normals(_N_x * _N_y);
  for (size_t m = 0; m < _N_x; ++m) {
    for (size_t n = 0; n < _N_y; ++n) {
      // index to any of the maps/images
      const size_t i = m + n * _N_x;

      // select the center in the original image
      const double c_x_orig = (double(width) / double(2 * _N_x))
          + m * (double(width) / double(_N_x));
      const double c_y_orig = (double(height) / double(2 * _N_y))
          + n * (double(height) / double(_N_y));

      // find the center in the 3D camera frame
      cameras::Point3d p_C_ci;
      cameraGeometryPtr->keypointToEuclidean(
          cameras::Point2d(c_x_orig, c_y_orig), p_C_ci);
      normals[i] = cv::normalize(p_C_ci);  // unit vector
    }
  }

  for (size_t m = 0; m < _N_x; ++m) {
    for (size_t n = 0; n < _N_y; ++n) {

      // index to any of the maps/images
      const size_t i = m + n * _N_x;

      // now we can also define and store the 3D rotation
      // that brings rays from the global camera frame C into the local frame Ci.
      cv::Matx33d R_Ci_C;
      cv::Rodrigues(normals[i].cross(cv::Vec3d(0, 0, 1)), R_Ci_C);
      cv::Matx33d R_C_Ci = R_Ci_C.t();

      // now determine the boundaries / image size. this is a bit annoying...
      // if the sub-image is at the image boundary, we need to trace it.
      bool leftBoundary = false;
      bool rightBoundary = false;
      bool topBoundary = false;
      bool bottomBoundary = false;
      if (m == 0)
        leftBoundary = true;
      if (m == _N_x - 1)
        rightBoundary = true;
      if (n == 0)
        topBoundary = true;
      if (n == _N_y - 1)
        bottomBoundary = true;
      // the following will be the corner points of the image:
      cv::Vec3d p_00(0, 0, 0);
      cv::Vec3d p_10(0, 0, 0);
      cv::Vec3d p_01(0, 0, 0);
      cv::Vec3d p_11(0, 0, 0);
      // the 3-plane intersections are easy, so do them first, if applicable:
      if (!leftBoundary && !topBoundary) {
        threePlaneIntersection(normals[i], -1.0, normals[i - 1], -1.0,
                               normals[i - _N_x],
                               -1.0, p_00);
        p_00 = R_Ci_C * p_00;
        //std::cout<<p_00[2]<<" ";
        p_00[0] /= p_00[2];
        p_00[2] = 1.0;
      }
      if (!topBoundary && !rightBoundary) {
        threePlaneIntersection(normals[i], -1.0, normals[i - _N_x], -1.0,
                               normals[i + 1],
                               -1.0, p_10);
        p_10 = R_Ci_C * p_10;
        //std::cout<<p_10[2]<<" ";
        p_10[0] /= p_10[2];
        p_10[2] = 1.0;
      }
      if (!leftBoundary && !bottomBoundary) {
        threePlaneIntersection(normals[i], -1.0, normals[i - 1], -1.0,
                               normals[i + _N_x],
                               -1.0, p_01);
        p_01 = R_Ci_C * p_01;
        //std::cout<<p_01[2]<<" ";
        p_01[0] /= p_01[2];
        p_01[2] = 1.0;
      }
      if (!rightBoundary && !bottomBoundary) {
        threePlaneIntersection(normals[i], -1.0, normals[i + 1], -1.0,
                               normals[i + _N_x],
                               -1.0, p_11);
        p_11 = R_Ci_C * p_11;
        p_11[0] /= p_11[2];
        //std::cout<<p_11[2]<<" ";
        p_11[2] = 1.0;
      }
      // trace left boundary TODO
      if (leftBoundary) {
        for (size_t y = 0; y < height; ++y) {
          cameras::Point3d p_C;
          cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(0, y), p_C);
          p_C = cv::normalize(p_C);
          cameras::Point3d p_Ci = R_Ci_C * p_C;
          p_Ci = p_Ci / p_Ci[2];
          if (!topBoundary && p_Ci[1] < std::min(p_00[1], p_10[1]))
            continue;  // outside the image
          if (!bottomBoundary && p_Ci[1] > std::max(p_01[1], p_11[1]))
            continue;  // outside the image
          if (p_Ci[0] < p_00[0]) {
            p_00[0] = p_Ci[0];
            p_01[0] = p_Ci[0];
          }
        }
      }
      // trace right boundary TODO
      if (rightBoundary) {
        for (size_t y = 0; y < height; ++y) {
          cameras::Point3d p_C;
          cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(width, y),
                                                 p_C);
          p_C = cv::normalize(p_C);
          cameras::Point3d p_Ci = R_Ci_C * p_C;
          p_Ci = p_Ci / p_Ci[2];
          if (!topBoundary && p_Ci[1] < std::min(p_00[1], p_10[1]))
            continue;  // outside the image
          if (!bottomBoundary && p_Ci[1] > std::max(p_01[1], p_11[1]))
            continue;  // outside the image
          if (p_Ci[0] > p_10[0]) {
            p_10[0] = p_Ci[0];
            p_11[0] = p_Ci[0];
          }
        }
      }
      // trace top boundary
      if (topBoundary) {
        for (size_t x = 0; x < width; ++x) {
          cameras::Point3d p_C;
          cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(x, 0), p_C);
          p_C = cv::normalize(p_C);
          cameras::Point3d p_Ci = R_Ci_C * p_C;
          p_Ci = p_Ci / p_Ci[2];
          if (!leftBoundary && p_Ci[0] < std::min(p_00[0], p_01[0]))
            continue;  // outside the image
          if (!rightBoundary && p_Ci[0] > std::max(p_10[0], p_11[0]))
            continue;  // outside the image
          if (p_Ci[1] < p_00[1]) {
            p_00[1] = p_Ci[1];
            p_10[1] = p_Ci[1];
          }
        }
      }
      // trace bottom boundary TODO
      if (bottomBoundary) {
        for (size_t x = 0; x < width; ++x) {
          cameras::Point3d p_C;
          cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(x, height),
                                                 p_C);
          p_C = cv::normalize(p_C);
          cameras::Point3d p_Ci = R_Ci_C * p_C;
          p_Ci = p_Ci / p_Ci[2];
          if (!leftBoundary && p_Ci[0] < std::min(p_00[0], p_01[0]))
            continue;  // outside the image
          if (!rightBoundary && p_Ci[0] > std::max(p_10[0], p_11[0]))
            continue;  // outside the image
          if (p_Ci[1] > p_01[1]) {
            p_01[1] = p_Ci[1];
            p_11[1] = p_Ci[1];
          }
        }
      }

      const size_t margin = 100;

      double imageCenterU = -std::min(p_00[0], p_01[0]) * n_focalLength;
      if (!leftBoundary)
        imageCenterU += margin;
      double imageCenterV = -std::min(p_00[1], p_10[1]) * n_focalLength;
      if (!topBoundary)
        imageCenterV += margin;
      int pixelsU = imageCenterU + std::max(p_10[0], p_11[0]) * n_focalLength;
      if (!rightBoundary)
        pixelsU += margin;
      int pixelsV = imageCenterV + std::max(p_01[1], p_11[1]) * n_focalLength;
      if (!bottomBoundary)
        pixelsV += margin;

      //std::cout<<p_00<<p_10<<p_01<<p_11<<std::endl;
      // creat the undistortion model
      _undistortedModels.push_back(
          cameras::PinholeCameraGeometry<cameras::NoDistortion>(n_focalLength,
                                                                n_focalLength,
                                                                imageCenterU,
                                                                imageCenterV,
                                                                pixelsU,
                                                                pixelsV));

      // init all maps
      _distort_x_maps[i] = cv::Mat::zeros(pixelsV, pixelsU, CV_32FC1);
      _distort_y_maps[i] = cv::Mat::zeros(pixelsV, pixelsU, CV_32FC1);
      _distort_1_maps[i] = cv::Mat::zeros(pixelsV, pixelsU, CV_16SC2);
      _distort_2_maps[i] = cv::Mat::zeros(pixelsV, pixelsU, CV_16UC1);
      _undistort_x_maps[i] = cv::Mat::zeros(height, width, CV_32FC1);
      _undistort_y_maps[i] = cv::Mat::zeros(height, width, CV_32FC1);
      _undistort_1_maps[i] = cv::Mat::zeros(height, width, CV_16SC2);
      _undistort_2_maps[i] = cv::Mat::zeros(height, width, CV_16UC1);

      // distortion maps
      for (int x = 0; x < pixelsU; ++x) {
        for (int y = 0; y < pixelsV; ++y) {
          // project to standardized plane
          cameras::Point2d kpt;
          cameraGeometryPtr->euclideanToKeypoint(
              R_C_Ci
                  * cameras::Point3d((double(x) - imageCenterU) / n_focalLength,
                                     (double(y) - imageCenterV) / n_focalLength,
                                     1.0),
              kpt);

          _distort_x_maps[i].at<float>(y, x) = kpt[0];
          _distort_y_maps[i].at<float>(y, x) = kpt[1];
        }
      }
      // convert to fix point maps
      cv::convertMaps(_distort_x_maps[i], _distort_y_maps[i],
                      _distort_1_maps[i],
                      _distort_2_maps[i], CV_16SC2);

      // undistortion maps
      for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
          // project to standardized plane
          cameras::Point3d p_C;
          cameraGeometryPtr->keypointToEuclidean(
              cameras::Point2d(double(x), double(y)), p_C);
          cameras::Point3d p_Ci = R_Ci_C * p_C;
          _undistort_x_maps[i].at<float>(y, x) = p_Ci[0] / p_Ci[2]
              * n_focalLength + imageCenterU;
          _undistort_y_maps[i].at<float>(y, x) = p_Ci[1] / p_Ci[2]
              * n_focalLength + imageCenterV;
        }
      }

      // convert to fix point maps
      cv::convertMaps(_undistort_x_maps[i], _undistort_y_maps[i],
                      _undistort_1_maps[i],
                      _undistort_2_maps[i], CV_16SC2);

      // store look-up
      cv::Mat img = cv::Mat::ones(pixelsV, pixelsU, CV_8UC1) * (i + 1);
      if (!leftBoundary)
        img.colRange(0, margin).setTo(0);
      if (!topBoundary)
        img.rowRange(0, margin).setTo(0);
      if (!rightBoundary)
        img.colRange(img.cols - margin, img.cols).setTo(0);
      if (!bottomBoundary)
        img.rowRange(img.rows - margin, img.rows).setTo(0);
      cv::Mat currentSelection = cv::Mat::zeros(pixelsV, pixelsU, CV_8UC1);
      cv::remap(img, currentSelection, _undistort_1_maps[i],
                _undistort_2_maps[i],
                cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      cv::max(_cameraModelSelection, currentSelection, _cameraModelSelection);
    }
  }
  //cv::imshow("allocation",_cameraModelSelection*(255/_undistort_x_maps.size()));
}

bool CameraAwareFeature::threePlaneIntersection(const cv::Vec3d& n1, double d1,
                                                const cv::Vec3d& n2,
                                                double d2,
                                                const cv::Vec3d& n3,
                                                double d3,
                                                cv::Vec3d& result) {

  double denom = n1.ddot(n2.cross(n3));

  if (fabs(denom) < 1e-12)
    return false;

  result = (n2.cross(n3) * d1 + n3.cross(n1) * d2 + n1.cross(n2) * d3)
      / (-denom);

  return true;
}

/*void CameraAwareFeature::setCameraGeometry(const cv::Ptr<cameras::CameraGeometryBase> cameraGeometryPtr){

 // store the geometry
 _cameraGeometryPtr = cameraGeometryPtr;

 // generate the maps

 // first, project corners to undistorted image, in order to determine its size
 // we assume that the corner rays are farthest apart
 cameras::Point3d p_C_00, p_C_w0, p_C_0h, p_C_wh;
 cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(0,0),p_C_00);
 cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(cameraGeometryPtr->width(),0),p_C_w0);
 cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(0,cameraGeometryPtr->height()),p_C_0h);
 cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(cameraGeometryPtr->width(),cameraGeometryPtr->height()),p_C_wh);
 p_C_00 = cv::normalize(p_C_00);
 p_C_w0 = cv::normalize(p_C_w0);
 p_C_0h = cv::normalize(p_C_0h);
 p_C_wh = cv::normalize(p_C_wh);

 // original image dimensions
 const size_t width = cameraGeometryPtr->width();
 const size_t height = cameraGeometryPtr->height();

 // figure out focal length
 cameras::Point3d p_C_mc, p_C_pc;
 cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(cameraGeometryPtr->width()/2.0-1.0,cameraGeometryPtr->height()/2.0),p_C_mc);
 cameraGeometryPtr->keypointToEuclidean(cameras::Point2d(cameraGeometryPtr->width()/2.0+1.0,cameraGeometryPtr->height()/2.0),p_C_pc);
 const double n_focalLength = 1.0/((p_C_pc[0]/p_C_pc[2]-p_C_mc[0]/p_C_mc[2])/2.0);

 const size_t n_width = (-std::min(p_C_00[0]/p_C_00[2],p_C_0h[0]/p_C_0h[2]) +
 std::max(p_C_w0[0]/p_C_w0[2],p_C_wh[0]/p_C_wh[2]))*n_focalLength +
 0.5;
 const size_t n_height = (-std::min(p_C_00[1]/p_C_00[2],p_C_w0[1]/p_C_w0[2]) +
 std::max(p_C_0h[1]/p_C_0h[2],p_C_wh[1]/p_C_wh[2]))*n_focalLength +
 0.5;

 const double c_x=double(-std::min(p_C_00[0]/p_C_00[2],p_C_0h[0]/p_C_0h[2]))*n_focalLength;
 const double c_y=double(-std::min(p_C_00[1]/p_C_00[2],p_C_w0[1]/p_C_w0[2]))*n_focalLength;

 // find out into how many different sub-images to split:
 const double angle_x=std::max(acos(p_C_00[0]*p_C_w0[0]+p_C_00[1]*p_C_w0[1]+p_C_00[2]*p_C_w0[2]),
 acos(p_C_0h[0]*p_C_wh[0]+p_C_0h[1]*p_C_wh[1]+p_C_0h[2]*p_C_wh[2]));
 const double angle_y=std::max(acos(p_C_00[0]*p_C_0h[0]+p_C_00[1]*p_C_0h[1]+p_C_00[2]*p_C_0h[2]),
 acos(p_C_w0[0]*p_C_wh[0]+p_C_w0[1]*p_C_wh[1]+p_C_w0[2]*p_C_wh[2]));
 //std::cout<<angle_x/M_PI*180<<std::endl;
 //std::cout<<angle_y/M_PI*180<<std::endl;
 _N_x = angle_x/2.0/_distortionTolerance+1.0;
 _N_y = angle_y/2.0/_distortionTolerance+1.0;
 std::cout<<_N_x<<","<<_N_y<<std::endl;

 // for now, support only one globally undistorted image:
 _distort_x_maps.resize(1);
 _distort_y_maps.resize(1);
 _distort_1_maps.resize(1);
 _distort_2_maps.resize(1);
 _undistort_x_maps.resize(1);
 _undistort_y_maps.resize(1);
 _undistort_1_maps.resize(1);
 _undistort_2_maps.resize(1);

 // init all maps
 _distort_x_maps[0] = cv::Mat::zeros(n_height, n_width, CV_32FC1);
 _distort_y_maps[0] = cv::Mat::zeros(n_height, n_width, CV_32FC1);
 _distort_1_maps[0] = cv::Mat::zeros(n_height, n_width, CV_16SC2);
 _distort_2_maps[0] = cv::Mat::zeros(n_height, n_width, CV_16UC1);
 _undistort_x_maps[0] = cv::Mat::zeros(height, width, CV_32FC1);
 _undistort_y_maps[0] = cv::Mat::zeros(height, width, CV_32FC1);
 _undistort_1_maps[0] = cv::Mat::zeros(height, width, CV_16SC2);
 _undistort_2_maps[0] = cv::Mat::zeros(height, width, CV_16UC1);

 // distortion
 for(size_t x = 0; x < n_width; ++x){
 for(size_t y = 0; y < n_height; ++y){
 // project to standardized plane
 cameras::Point2d kpt;
 cameraGeometryPtr->euclideanToKeypoint(
 cameras::Point3d((double(x)-c_x)/n_focalLength,(double(y)-c_y)/n_focalLength,1.0),
 kpt);

 _distort_x_maps[0].at<float>(y,x)=kpt[0];
 _distort_y_maps[0].at<float>(y,x)=kpt[1];
 }
 }

 // convert to fix point maps
 cv::convertMaps(_distort_x_maps[0],_distort_y_maps[0],_distort_1_maps[0],_distort_2_maps[0], CV_16SC2);

 // undistortion:
 for(size_t x = 0; x < width; ++x){
 for(size_t y = 0; y < height; ++y){
 // project to standardized plane
 cameras::Point3d p_C;
 cameraGeometryPtr->keypointToEuclidean(
 cameras::Point2d(double(x),double(y)),p_C);
 _undistort_x_maps[0].at<float>(y,x)=p_C[0]/p_C[2]*n_focalLength+c_x;
 _undistort_y_maps[0].at<float>(y,x)=p_C[1]/p_C[2]*n_focalLength+c_y;
 }
 }

 // convert to fix point maps
 cv::convertMaps(_undistort_x_maps[0],_undistort_y_maps[0],_undistort_1_maps[0],_undistort_2_maps[0], CV_16SC2);

 }*/

/* cv::Feature2d  interface */
void CameraAwareFeature::detectAndCompute(cv::InputArray image,
                                          cv::InputArray mask,
                                          std::vector<cv::KeyPoint>& keypoints,
                                          cv::OutputArray descriptors,
                                          bool useProvidedKeypoints) {

  /*for(size_t i=0; i<_N_x*_N_y; ++i){
   cv::Mat undistorted;
   cv::remap(image,undistorted,_distort_1_maps[i],_distort_2_maps[i],cv::INTER_LINEAR);
   std::stringstream windowname;
   windowname<<"undistorted "<<i<<std::endl;
   cv::imshow(windowname.str(),undistorted);
   }
   return;*/

  // convert mask - currently not needed
  //cv::Mat undistorted_mask;
  //if(!mask.empty())
  //cv::remap(mask,undistorted_mask,_distort_1_maps[0],_distort_2_maps[0],cv::INTER_LINEAR);
  /*std::stringstream windowname;
   windowname<<"undistorted "<< (size_t)this;
   cv::imshow(windowname.str(),undistorted_image);*/

  // do it.
  // handle provided keypoints correctly
  if (!useProvidedKeypoints) {
    keypoints.clear();  // TODO: if keypoints are provided, they need undistortion...
  }

  if (image.empty())
    return;

  CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));

  // detection
  if (useProvidedKeypoints) {
    brisk::BriskFeature* briskFeaturePtr =
        dynamic_cast<brisk::BriskFeature*>(_feature2dPtr.get());
    if (briskFeaturePtr)
      briskFeaturePtr->detect(image, keypoints, mask);  // this is already taking keypoints, if provided
  } else {
    _feature2dPtr->detect(image, keypoints, mask);  // this will clear keypoints
  }

  // remove boundary points
  removeBorderKeypoints(2.0, image.getMat(), keypoints);

  // group the keypoints / descriptors
  std::vector < std::vector<cv::KeyPoint> > keypointsVec(_N_x * _N_y);
  std::vector<cv::Mat> descriptorsVec(_N_x * _N_y);
  int original_class_id = -1;
  int nrows = _cameraModelSelection.rows;
  int ncols = _cameraModelSelection.cols;
  for (size_t k = 0; k < keypoints.size(); ++k) {
	  int x = std::rint(keypoints[k].pt.x);
	  int y = std::rint(keypoints[k].pt.y);
	  CV_Assert((x >= 0) && (x < ncols)); // todo: change this to debug assert
	  CV_Assert((y >= 0) && (y < nrows)); // todo: change this to debug assert
    size_t idx = (_cameraModelSelection.at<uchar>(y, x));
    if (idx == 0)
      continue;  // meaning there is no image assigned... maybe we should issue a warning here...?
    idx -= 1;

    if (original_class_id == -1)
    {
      original_class_id = keypoints[k].class_id;
    }
    keypoints[k].class_id = k;  // abuse the class id
    keypointsVec.at(idx).push_back(keypoints[k]);
  }

  // go through the groups and compute the descriptors
  size_t numFeatures = 0;
  std::vector<cv::KeyPoint> keypoints_tmp;
  keypoints_tmp.swap(keypoints);
  keypoints.clear();
  keypoints.reserve(keypoints_tmp.size());
  for (size_t i = 0; i < keypointsVec.size(); ++i)
  {
	size_t numKeypointsInThisClass = keypointsVec[i].size();

    // remap image
    cv::Mat undistorted_image;
    cv::remap(image, undistorted_image, _distort_1_maps[i], _distort_2_maps[i], cv::INTER_LINEAR);

    // undistortion
    std::vector<cv::KeyPoint> undistortedKeypoints;
    undistortKeypoints(i, keypointsVec[i], undistortedKeypoints);
    std::vector<cv::KeyPoint> undistortedKeypoints_bkp = undistortedKeypoints;  // back up, since compute might secretly delete...

    // check: there should be as many undistorted keypoints as there were intial keypoints of this class
    size_t numUndistortedKeypointsInThisClass = undistortedKeypoints.size();
    CV_Assert(numUndistortedKeypointsInThisClass == numKeypointsInThisClass);

    // override extraction direction, if requested:
    double e_C_dotprod = _e_C.dot(_e_C);
    if(e_C_dotprod > 0){
      for(size_t k=0; k<undistortedKeypoints.size(); ++k){
        // FIXME: this is still very dumb and inefficient, could be done wiht look-ups...
        cameras::Vec2d e_orig;
        cameras::Point3d p_C;
        cameras::Matx23d J_23;
        cameras::Point2d kp_orig_dummy;
        cameras::Point2d kp_orig(keypointsVec[i][k].pt.x,keypointsVec[i][k].pt.y);
        _cameraGeometryPtr->keypointToEuclidean(kp_orig,p_C);
        _cameraGeometryPtr->euclideanToKeypoint(p_C,kp_orig_dummy,J_23);
        cameras::Vec2d e_kp_orig = J_23*_e_C;
        double length = sqrt(e_kp_orig.dot(e_kp_orig));
        if(length<0.1)
          continue; // leave original angle, or, to be set in the descriptor extraction as BRISK will...
        e_kp_orig=e_kp_orig*1.0/length; // normalize...
        //std::cout<<atan2(e_kp_orig[1],e_kp_orig[0])*180.0/M_PI<<std::endl;
        cameras::Point2d kp_orig_2 = kp_orig+keypointsVec[i][k].size*e_kp_orig;
        cameras::Point2d kp_2;
        undistortPoint(i,kp_orig_2,kp_2);
        cameras::Vec2d e_kp = kp_2-cameras::Point2d(undistortedKeypoints[k].pt.x,undistortedKeypoints[k].pt.y);
        //std::cout<<kp_orig.t()<<" "<<kp_orig_2.t()<<std::endl;
        undistortedKeypoints[k].angle=atan2(e_kp[1],e_kp[0])*180.0/M_PI;
      }
    }

    // extraction - on undistorted image
    cv::Mat descriptors_;
    //CV_Assert(undistortedKeypoints.size() > 0); //todo: change this to CV_DbgAssert()

    _feature2dPtr->compute(undistorted_image, undistortedKeypoints, descriptors_);
    //distortKeypoints(undistortedKeypoints,keypoints);

    // check! there descriptors_ should have as many rows as there were intially keypoints in this class!
    numKeypointsInThisClass = undistortedKeypoints.size();
    CV_Assert(static_cast<size_t>(descriptors_.rows) == numKeypointsInThisClass);

    descriptorsVec[i] = descriptors_;  // convert input output array

    /*cv::Mat out;
    cv::drawKeypoints(undistorted_image,undistortedKeypoints,out,cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("undistorted_image",out);
    cv::waitKey();*/

    numFeatures += undistortedKeypoints.size();

    // also restore keypoints output
    for (size_t k = 0; k < undistortedKeypoints.size(); ++k) {
      const cv::KeyPoint& kp=undistortedKeypoints[k];
      keypoints.push_back(keypoints_tmp[undistortedKeypoints[k].class_id]);
      keypoints.back().class_id = original_class_id;
      // transform angle
      cameras::Point2d ptkp(kp.pt.x, kp.pt.y);
      cameras::Point2d ptkp_orig(keypoints.back().pt.x, keypoints.back().pt.y);
      cameras::Vec2d e_kp(cos(kp.angle/180.0*M_PI),sin(kp.angle/180.0*M_PI));
      cameras::Point2d ptkp_2=ptkp+kp.size*e_kp; // use the size to equalize nonlinearity
      cameras::Point2d ptkp_2_orig;
      distortPoint(i,ptkp_2,ptkp_2_orig);
      cameras::Vec2d e_kp_orig = ptkp_2_orig-ptkp_orig;
      keypoints.back().angle=atan2(e_kp_orig[1],e_kp_orig[0])*180.0/M_PI;
      //std::cout<<keypoints.back().angle<<" "<<kp.angle<<std::endl;
      //std::cout<<atan2(e_kp[1],e_kp[0])*180.0/M_PI<<" "<<kp.angle<<std::endl;
    }
  }

  // assemble descriptor output
  if (descriptorsVec.size() == 0)
    return;  // would be very weird...

  const int numBriskBytes = brisk::BriskDescriptorExtractor::kDescriptorLength / 8;
  cv::Mat descriptors_final(static_cast<int>(numFeatures), numBriskBytes, 0);
  size_t start_row = 0;
  // I believe here we should no iterate over the keypointsVec, because featured2d->compute(...) might have thrown away some keypoints
  // for these, there no descriptors and hence there's no equivalence anymore between the descriptorVec and the keypointsVec!
  for (size_t i = 0; i < descriptorsVec.size(); ++i)
  {
	size_t nrows = static_cast<size_t>(descriptorsVec.at(i).rows);
    if (nrows > 0)
    {
      size_t end = start_row + nrows;
      CV_Assert(end <= numFeatures); //todo: change to debug assert
      descriptorsVec.at(i).copyTo(descriptors_final.rowRange(start_row, end));
      start_row += nrows;
    }
  }
  descriptors.getMatRef() = descriptors_final;

  //std::cout<<descriptors.getMat().rows<<std::endl;
  //std::cout<<undistortedKeypoints.size()<<" vs "<<undistortedKeypoints_bkp.size()<<std::endl;
}

/* cv::FeatureDetector interface */
void CameraAwareFeature::detectImpl(const cv::Mat& image,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    const cv::Mat& mask) const {

  // attention: using this won't allow using passed keypoints...

  // run detection
  _feature2dPtr->detect(image, keypoints, mask);

  // remove boundary points
  removeBorderKeypoints(2.0, image, keypoints);
}

void CameraAwareFeature::distortPoint(
    size_t modelIdx,
    const cameras::Point2d& point_undistorted_in,
    cameras::Point2d& point_distorted_out) const{
  // bilinear interpolation :
  const size_t x_i = size_t(point_undistorted_in[0]);  // floor x
  const float r_x = (point_undistorted_in[0] - x_i);  // ratio x
  const size_t y_i = size_t(point_undistorted_in[1]);  // floor y
  const float r_y = (point_undistorted_in[01] - y_i);  // ratio y

  cv::Point2f p_00(_distort_x_maps[modelIdx].at<float>(y_i, x_i),
                  _distort_y_maps[modelIdx].at<float>(y_i, x_i));
  cv::Point2f p_10(_distort_x_maps[modelIdx].at<float>(y_i, x_i + 1),
                  _distort_y_maps[modelIdx].at<float>(y_i, x_i + 1));
  cv::Point2f p_01(_distort_x_maps[modelIdx].at<float>(y_i + 1, x_i),
                  _distort_y_maps[modelIdx].at<float>(y_i + 1, x_i));
  cv::Point2f p_11(_distort_x_maps[modelIdx].at<float>(y_i + 1, x_i + 1),
                  _distort_y_maps[modelIdx].at<float>(y_i + 1, x_i + 1));

  // bilinear interpolation
  cv::Point2f p_x0 = (p_00 + r_x * (p_10 - p_00));
  cv::Point2f p_x1 = (p_01 + r_x * (p_11 - p_01));

  cv::Point2f p_out = p_x0 + r_y * (p_x1 - p_x0);
  point_distorted_out=cameras::Point2d(p_out.x,p_out.y);
}

void CameraAwareFeature::undistortPoint(
    size_t modelIdx,
    const cameras::Point2d& point_distorted_in,
    cameras::Point2d& point_undistorted_out) const{
  // bilinear interpolation :
  const size_t x_i = size_t(point_distorted_in[0]);  // floor x
  const float r_x = (point_distorted_in[0] - x_i);  // ratio x
  const size_t y_i = size_t(point_distorted_in[1]);  // floor y
  const float r_y = (point_distorted_in[1] - y_i);  // ratio y

  cv::Point2f p_00(_undistort_x_maps[modelIdx].at<float>(y_i, x_i),
                   _undistort_y_maps[modelIdx].at<float>(y_i, x_i));
  cv::Point2f p_10(_undistort_x_maps[modelIdx].at<float>(y_i, x_i + 1),
                   _undistort_y_maps[modelIdx].at<float>(y_i, x_i + 1));
  cv::Point2f p_01(_undistort_x_maps[modelIdx].at<float>(y_i + 1, x_i),
                   _undistort_y_maps[modelIdx].at<float>(y_i + 1, x_i));
  cv::Point2f p_11(_undistort_x_maps[modelIdx].at<float>(y_i + 1, x_i + 1),
                   _undistort_y_maps[modelIdx].at<float>(y_i + 1, x_i + 1));

  // bilinear interpolation
  cv::Point2f p_x0 = (p_00 + r_x * (p_10 - p_00));
  cv::Point2f p_x1 = (p_01 + r_x * (p_11 - p_01));

  cv::Point2f p_out = p_x0 + r_y * (p_x1 - p_x0);
  point_undistorted_out=cameras::Point2d(p_out.x,p_out.y);
}


void CameraAwareFeature::distortKeypoints(
    size_t modelIdx, const std::vector<cv::KeyPoint>& keypoints_undistorted_in,
    std::vector<cv::KeyPoint>& keypoints_distorted_out) const {

  keypoints_distorted_out.clear();
  keypoints_distorted_out.reserve(keypoints_undistorted_in.size());
  for (size_t k = 0; k < keypoints_undistorted_in.size(); ++k) {

    // bilinear interpolation :
    const size_t x_i = size_t(keypoints_undistorted_in[k].pt.x);  // floor x
    const float r_x = (keypoints_undistorted_in[k].pt.x - x_i);  // ratio x
    const size_t y_i = size_t(keypoints_undistorted_in[k].pt.y);  // floor y
    const float r_y = (keypoints_undistorted_in[k].pt.y - y_i);  // ratio y

    cv::Point2f p_00(_distort_x_maps[modelIdx].at<float>(y_i, x_i),
                     _distort_y_maps[modelIdx].at<float>(y_i, x_i));
    cv::Point2f p_10(_distort_x_maps[modelIdx].at<float>(y_i, x_i + 1),
                     _distort_y_maps[modelIdx].at<float>(y_i, x_i + 1));
    cv::Point2f p_01(_distort_x_maps[modelIdx].at<float>(y_i + 1, x_i),
                     _distort_y_maps[modelIdx].at<float>(y_i + 1, x_i));
    cv::Point2f p_11(_distort_x_maps[modelIdx].at<float>(y_i + 1, x_i + 1),
                     _distort_y_maps[modelIdx].at<float>(y_i + 1, x_i + 1));

    // bilinear interpolation
    cv::Point2f p_x0 = (p_00 + r_x * (p_10 - p_00));
    cv::Point2f p_x1 = (p_01 + r_x * (p_11 - p_01));

    keypoints_distorted_out.push_back(keypoints_undistorted_in[k]);
    keypoints_distorted_out.back().pt = p_x0 + r_y * (p_x1 - p_x0);
  }
}

void CameraAwareFeature::undistortKeypoints(
    size_t modelIdx, const std::vector<cv::KeyPoint>& keypoints_distorted_in,
    std::vector<cv::KeyPoint>& keypoints_undistorted_out) const {

  keypoints_undistorted_out.clear();
  keypoints_undistorted_out.reserve(keypoints_distorted_in.size());
  for (size_t k = 0; k < keypoints_distorted_in.size(); ++k) {

    // bilinear interpolation :
    const size_t x_i = size_t(keypoints_distorted_in[k].pt.x);  // floor x
    const float r_x = (keypoints_distorted_in[k].pt.x - x_i);  // ratio x
    const size_t y_i = size_t(keypoints_distorted_in[k].pt.y);  // floor y
    const float r_y = (keypoints_distorted_in[k].pt.y - y_i);  // ratio y

    cv::Point2f p_00(_undistort_x_maps[modelIdx].at<float>(y_i, x_i),
                     _undistort_y_maps[modelIdx].at<float>(y_i, x_i));
    cv::Point2f p_10(_undistort_x_maps[modelIdx].at<float>(y_i, x_i + 1),
                     _undistort_y_maps[modelIdx].at<float>(y_i, x_i + 1));
    cv::Point2f p_01(_undistort_x_maps[modelIdx].at<float>(y_i + 1, x_i),
                     _undistort_y_maps[modelIdx].at<float>(y_i + 1, x_i));
    cv::Point2f p_11(_undistort_x_maps[modelIdx].at<float>(y_i + 1, x_i + 1),
                     _undistort_y_maps[modelIdx].at<float>(y_i + 1, x_i + 1));

    // bilinear interpolation
    cv::Point2f p_x0 = (p_00 + r_x * (p_10 - p_00));
    cv::Point2f p_x1 = (p_01 + r_x * (p_11 - p_01));

    keypoints_undistorted_out.push_back(keypoints_distorted_in[k]);
    keypoints_undistorted_out.back().pt = p_x0 + r_y * (p_x1 - p_x0);
  }
}

// remove points too close to boundary:
void CameraAwareFeature::removeBorderKeypoints(
    double scale, const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints) const {
  for (size_t k = 0; k < keypoints.size();) {
    if ((keypoints[k].pt.x - keypoints[k].size * scale < 0.0)
        || (keypoints[k].pt.y - keypoints[k].size * scale < 0.0)
        || (keypoints[k].pt.x + keypoints[k].size * scale > float(image.cols))
        || (keypoints[k].pt.y + keypoints[k].size * scale > float(image.rows))) {
      keypoints.erase(keypoints.begin() + k);
    } else {
      ++k;
    }
  }
}

/* cv::DescriptorExtractor interface */
void CameraAwareFeature::computeImpl(const cv::Mat& /*image*/,
                                     std::vector<cv::KeyPoint>& /*keypoints*/,
                                     cv::Mat& /*descriptors*/) const {
  // convert image and mask
  /*cv::Mat undistorted_image, undistorted_mask;
   cv::remap(image,undistorted_image,_distort_1_maps[0],_distort_2_maps[0],cv::INTER_LINEAR);

   // run extraction
   _feature2dPtr->compute(undistorted_image, keypoints, descriptors);

   // convert back:
   distortKeypoints(keypoints);*/
}

}  // namespace brisk

