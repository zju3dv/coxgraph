/*
 * CameraAwareFeature.hpp
 *
 *  Created on: Dec 24, 2013
 *      Author: lestefan
 */

#ifndef CAMERAAWAREFEATURE_HPP_
#define CAMERAAWAREFEATURE_HPP_

#include <opencv2/features2d/features2d.hpp>
#include <brisk/cameras/cameras.h>

namespace brisk {

class CameraAwareFeature : public cv::Feature2D {
 public:

  CameraAwareFeature();
  CameraAwareFeature(
      cv::Ptr<cv::Feature2D> feature2dPtr,
      const/*CAMERA_GEOMETRY_T*/cv::Ptr<cameras::CameraGeometryBase> cameraGeometryPtr,
      double distortionTolerance = 2e-1);
  ~CameraAwareFeature();

  // setters
  void setFeature2d(cv::Ptr<cv::Feature2D> feature2dPtr);
  //template<class CAMERA_GEOMETRY_T>
  void setCameraGeometry(
      const/*CAMERA_GEOMETRY_T*/cv::Ptr<cameras::CameraGeometryBase> cameraGeometryPtr);
  void setDistortionTolerance(double distortionTolerance);
  void setExtractionDirection(const cameras::Vec3d& e_C) {
    _e_C=e_C*1.0/sqrt(e_C.dot(e_C)); // set normalized...
  }

  /* cv::Feature2d  interface */
  virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::OutputArray descriptors,
                                bool useProvidedKeypoints = false);

  /* cv::DescriptorExtractor interface */
  virtual int descriptorSize() const {
    return _feature2dPtr->descriptorSize();
  }
  virtual int descriptorType() const {
    return _feature2dPtr->descriptorType();
  }

 protected:

  /* cv::FeatureDetector interface */
  virtual void detectImpl(const cv::Mat& image,
                          std::vector<cv::KeyPoint>& keypoints,
                          const cv::Mat& mask = cv::Mat()) const;

  /* cv::DescriptorExtractor interface */
  virtual void computeImpl(const cv::Mat& image,
                           std::vector<cv::KeyPoint>& keypoints,
                           cv::Mat& descriptors) const;

  // contains the underlying feature detection and extraction
  cv::Ptr<cv::Feature2D> _feature2dPtr;

  // specifies the distortion tolerance w.r.t. perspective projection
  double _distortionTolerance;

  // these are the distort and undistort maps
  std::vector<cv::Mat> _distort_x_maps;
  std::vector<cv::Mat> _distort_y_maps;
  std::vector<cv::Mat> _distort_1_maps;
  std::vector<cv::Mat> _distort_2_maps;
  std::vector<cv::Mat> _undistort_x_maps;
  std::vector<cv::Mat> _undistort_y_maps;
  std::vector<cv::Mat> _undistort_1_maps;
  std::vector<cv::Mat> _undistort_2_maps;

  size_t _N_x;
  size_t _N_y;

  cv::Mat _cameraModelSelection;

  cv::Ptr<cameras::CameraGeometryBase> _cameraGeometryPtr;

  // extraction direction
  cameras::Point3d _e_C;

  // camera models
  std::vector<cameras::PinholeCameraGeometry<cameras::NoDistortion> > _undistortedModels;

 private:
  void distortPoint(
      size_t modelIdx,
      const cameras::Point2d& point_undistorted_in,
      cameras::Point2d& point_distorted_out) const;
  void undistortPoint(
      size_t modelIdx,
      const cameras::Point2d& point_distorted_in,
      cameras::Point2d& point_undistorted_out) const;
  void distortKeypoints(
      size_t modelIdx,
      const std::vector<cv::KeyPoint>& keypoints_undistorted_in,
      std::vector<cv::KeyPoint>& keypoints_distorted_out) const;
  void undistortKeypoints(
      size_t modelIdx,
      const std::vector<cv::KeyPoint>& keypoints_undistorted_in,
      std::vector<cv::KeyPoint>& keypoints_distorted_out) const;

  void removeBorderKeypoints(double scale, const cv::Mat& image,
                             std::vector<cv::KeyPoint>& keypoints) const;

  static bool threePlaneIntersection(const cv::Vec3d& n1, double d1,
                                     const cv::Vec3d& n2, double d2,
                                     const cv::Vec3d& n3, double d3,
                                     cv::Vec3d& result);
};

}  // namespace brisk

#endif /* CAMERAAWAREFEATURE_HPP_ */
