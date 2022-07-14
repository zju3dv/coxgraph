/*
 * testCameras.cc
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#include <brisk/brisk.h>
#include <brisk/brute-force-matcher.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main(int /*argc*/, char ** /*argv*/) {
  // Process command line args.
  bool draw = false;
  bool tryOutExtractionDirection = false;

  enum CameraGeometryChoice 
  {
    EQUIDIST,
    RADIAL,
    UNDIST
  };
  
  CameraGeometryChoice camGeomtry = CameraGeometryChoice::EQUIDIST;
  
  // TODO (lestefan): make a proper unit test of this stuff...
  
  brisk::cameras::Point2d keypoint;
  brisk::cameras::Point3d p_C(0.1, 0.2, 1.0);
  p_C = p_C * 1.0 / cv::norm(p_C);  // normalize for later comparison
  brisk::cameras::Point3d p_C2;

  switch(camGeomtry)
  {
    case EQUIDIST:
      {
        brisk::cameras::EquidistantPinholeCameraGeometry cameraGeometry(602, 601, 318, 241, 640, 480, 
            brisk::cameras::EquidistantDistortion(0.3, 0.2, 0.01, 0.0002));
            
        cameraGeometry.euclideanToKeypoint(p_C, keypoint);
        cameraGeometry.keypointToEuclidean(keypoint, p_C2);
        break;
      }
    case RADIAL:
      {
        brisk::cameras::RadialTangentialPinholeCameraGeometry cameraGeometry(602, 601, 318, 241, 640, 480, brisk::cameras::RadialTangentialDistortion(0.3,0.2,0.01,0.0002));
        cameraGeometry.euclideanToKeypoint(p_C, keypoint);
        cameraGeometry.keypointToEuclidean(keypoint, p_C2);
        break;
      }
    case UNDIST:
      {
        brisk::cameras::UndistortedPinholeCameraGeometry cameraGeometry(602, 601, 318, 241, 640, 480);
        cameraGeometry.euclideanToKeypoint(p_C, keypoint);
        cameraGeometry.keypointToEuclidean(keypoint, p_C2);
        break;
      }
  }

  p_C2 = p_C2 * 1.0 / cv::norm(p_C2);  // normalize for later comparison
  std::cout << "original point:" << p_C << std::endl;
  std::cout << "projected and reprojected point:" << p_C2 << std::endl;
  std::cout << "difference:" << p_C2 - p_C << std::endl;

  // test the camera aware features

  cv::Ptr<brisk::cameras::CameraGeometryBase> cameraGeometry0Ptr =
      new brisk::cameras::RadialTangentialPinholeCameraGeometry(
          465.2005090331355,
          465.4821196969644,
          407.7552059925612,
          244.36152062408814,
          752,
          480,
          brisk::cameras::RadialTangentialDistortion(-0.3091674142711387,
                                                     0.10905899434862235,
                                                     9.527033720237582e-05,
                                                     -0.0005776582308113238));
  cv::Ptr<brisk::cameras::CameraGeometryBase> cameraGeometry1Ptr =
      new brisk::cameras::RadialTangentialPinholeCameraGeometry(
          466.13901328497104,
          466.5291851440462,
          358.83335820698255,
          250.46035698740574,
          752,
          480,
          brisk::cameras::RadialTangentialDistortion(-0.30098670704037306,
                                                     0.09347840167059357,
                                                     0.00020272056432327966,
                                                     -0.000576898338628004));

  cv::Ptr<cv::Feature2D> briskFeaturePtr = new brisk::BriskFeature(
      2, 30.0, 200.0, 400, true, true,
      brisk::BriskDescriptorExtractor::Version::briskV2);

  brisk::CameraAwareFeature camera0AwareFeature(briskFeaturePtr,
                                                cameraGeometry0Ptr, 2e-1);
  brisk::CameraAwareFeature camera1AwareFeature(briskFeaturePtr,
                                                cameraGeometry1Ptr, 2e-1);

  // try out the extraction direction...
  if(tryOutExtractionDirection)
  {
    camera0AwareFeature.setExtractionDirection(brisk::cameras::Vec3d(0,1,0));
    camera1AwareFeature.setExtractionDirection(brisk::cameras::Vec3d(0,1,0));
  }

  // read imgates
  cv::Mat img0, img1;
  img0 = cv::imread("../images/img0_aslam.pgm", cv::IMREAD_GRAYSCALE);
  img1 = cv::imread("../images/img1_aslam.pgm", cv::IMREAD_GRAYSCALE);
  if(draw)
  {
    cv::imshow("img0_aslam.pgm",img0);
    cv::imshow("img1_aslam.pgm",img1);
    cv::waitKey();
  }

  // detect and extract
  std::vector<cv::KeyPoint> kpts0;
  cv::Mat descriptors0;
  camera0AwareFeature.detectAndCompute(img0, cv::Mat(), kpts0, descriptors0);
  std::vector<cv::KeyPoint> kpts1;
  cv::Mat descriptors1;
  camera1AwareFeature.detectAndCompute(img1, cv::Mat(), kpts1, descriptors1);

  // match
  std::vector<std::vector<cv::DMatch> > matches;
  brisk::BruteForceMatcherSse matcher;
  matcher.radiusMatch(descriptors0, descriptors1, matches, 50.0);

  // draw stuff
  if(draw)
  {
   cv::Mat out0, out1;
   cv::drawKeypoints(img0,kpts0,out0);
   cv::drawKeypoints(img1,kpts1,out1);
   cv::imshow("img0_aslam.pgm",out0);
   cv::imshow("img1_aslam.pgm",out1);
  }
  
  cv::Mat out;
  cv::drawMatches(img0, kpts0, img1, kpts1, matches, out, cv::Scalar(0, 255, 0),
                  cv::Scalar(0, 0, 255), std::vector<std::vector<char> >(),
                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("matches", out);

  // do the thing again with conventional feature:
  kpts0.clear();
  descriptors0.setTo(0);
  briskFeaturePtr->detectAndCompute(img0, cv::Mat(), kpts0, descriptors0);
  kpts1.clear();
  descriptors1.setTo(0);
  briskFeaturePtr->detectAndCompute(img1, cv::Mat(), kpts1, descriptors1);

  // match
  std::vector<std::vector<cv::DMatch> > matches2;
  matcher.radiusMatch(descriptors0, descriptors1, matches2, 50.0);

  cv::Mat out2;
  cv::drawMatches(img0, kpts0, img1, kpts1, matches2, out2,
                  cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                  std::vector<std::vector<char> >(),
                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("matches conventional", out2);

  // display and wait
  cv::waitKey();

  return 0;

}

