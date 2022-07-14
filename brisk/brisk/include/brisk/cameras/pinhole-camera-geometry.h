/*
 * PinholeCameraGeometry.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#ifndef PINHOLECAMERAGEOMETRY_HPP_
#define PINHOLECAMERAGEOMETRY_HPP_

#include <brisk/cameras/camera-geometry-base.h>

namespace brisk {
namespace cameras {

template<class DISTORTION_T>
class PinholeCameraGeometry : public CameraGeometryBase {
 public:
  typedef DISTORTION_T distortion_t;

  virtual ~PinholeCameraGeometry() {
  }

  PinholeCameraGeometry();
  // construct the pinhole camera geometry with provided parameters
  PinholeCameraGeometry(double focalLengthU, double focalLengthV,
                        double imageCenterU, double imageCenterV, int pixelsU,
                        int pixelsV, const distortion_t& distortion =
                            distortion_t());

  // setters
  void setFocalLengthU(double focalLengthU) {
    _focalLengthU = focalLengthU;
    _recip_fu = 1.0 / _focalLengthU;
  }
  void setFocalLengthV(double focalLengthV) {
    _focalLengthV = focalLengthV;
    _recip_fv = 1.0 / _focalLengthV;
  }
  void setImageCenterU(double imageCenterU) {
    _imageCenterU = imageCenterU;
  }
  void setImageCenterV(double imageCenterV) {
    _imageCenterV = imageCenterV;
  }
  void setPixelsU(int pixelsU) {
    _pixelsU = pixelsU;
  }
  void setPixelsV(int pixelsV) {
    _pixelsV = pixelsV;
  }
  void setDistortion(const distortion_t& distortion) {
    _distortion = distortion;
  }

  // getters
  double focalLengthU() const {
    return _focalLengthU;
  }
  double focalLengthV() const {
    return _focalLengthV;
  }
  double imageCenterU() const {
    return _imageCenterU;
  }
  double imageCenterV() const {
    return _imageCenterV;
  }
  int pixelsU() const {
    return _pixelsU;
  }
  int pixelsV() const {
    return _pixelsV;
  }
  const distortion_t& distortion() const {
    return _distortion;
  }

  // inherited interfaces:

  // world-to-cam: return if the projection is valid and inside the image
  virtual bool euclideanToKeypoint(const Point3d& p_C_in,
                                   Point2d& point_out) const;
  virtual bool euclideanToKeypoint(const Point3d& p_C_in, Point2d& point_out,
                                   Matx23d& jacobian_out) const;
  virtual bool homogeneousToKeypoint(const HPoint4d& hp_C_in,
                                     Point2d& point_out) const;
  virtual bool homogeneousToKeypoint(const HPoint4d& hp_C_in,
                                     Point2d& point_out,
                                     Matx24d& jacobian_out) const;

  // cam-to-world: return if the inverse projection is valid
  virtual bool keypointToEuclidean(const Point2d& point_in,
                                   Point3d& p_C_out) const;
  virtual bool keypointToEuclidean(const Point2d& point_in, Point3d& p_C_out,
                                   Matx32d& inverse_jacobian_out) const;
  virtual bool keypointToHomogeneous(const Point2d& point_in,
                                     HPoint4d& hp_C_out) const;
  virtual bool keypointToHomogeneous(const Point2d& point_in,
                                     HPoint4d& hp_C_out,
                                     Matx42d& inverse_jacobian_out) const;

  virtual bool isValid(const Point2d& point_in) const;

  // get the width and height of the image as supported
  virtual size_t width() const;
  virtual size_t height() const;

 private:
  // geometry paramters
  double _focalLengthU;
  double _focalLengthV;
  double _imageCenterU;
  double _imageCenterV;
  size_t _pixelsU;
  size_t _pixelsV;

  // A computed value for speeding up computation.
  double _recip_fu;
  double _recip_fv;

  // distortion object
  distortion_t _distortion;

};

}  // namespace cameras
}  // namespace brisk

#include "implementation/pinhole-camera-geometry.h"

#endif /* PINHOLECAMERAGEOMETRY_HPP_ */
