/*
 * PinholeCameraGeometry.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

namespace brisk {
namespace cameras {

template<class DISTORTION_T>
PinholeCameraGeometry<DISTORTION_T>::PinholeCameraGeometry()
    : _focalLengthU(0.0),
      _focalLengthV(0.0),
      _imageCenterU(0.0),
      _imageCenterV(0.0),
      _pixelsU(0),
      _pixelsV(0),
      _recip_fu(0),
      _recip_fv(0) {
}

template<class DISTORTION_T>
PinholeCameraGeometry<DISTORTION_T>::PinholeCameraGeometry(
    double focalLengthU, double focalLengthV, double imageCenterU,
    double imageCenterV, int pixelsU, int pixelsV,
    const distortion_t& distortion)
    : _focalLengthU(focalLengthU),
      _focalLengthV(focalLengthV),
      _imageCenterU(imageCenterU),
      _imageCenterV(imageCenterV),
      _pixelsU(pixelsU),
      _pixelsV(pixelsV),
      _recip_fu(0),
      _recip_fv(0),
      _distortion(distortion) {
}

// world-to-cam: return if the projection is valid and inside the image
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::euclideanToKeypoint(
    const Point3d& p_C_in, Point2d& point_out) const {
  double rz = 1.0 / p_C_in[2];
  point_out[0] = p_C_in[0] * rz;
  point_out[1] = p_C_in[1] * rz;

  _distortion.distort(point_out);

  point_out[0] = _focalLengthU * point_out[0] + _imageCenterU;
  point_out[1] = _focalLengthV * point_out[1] + _imageCenterV;

  return isValid(point_out) && p_C_in[2] > 0;
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::euclideanToKeypoint(
    const Point3d& p_C_in, Point2d& point_out, Matx23d& jacobian_out) const {
  double rz = 1.0 / p_C_in[2];
  double rz2 = rz * rz;

  point_out[0] = p_C_in[0] * rz;
  point_out[1] = p_C_in[1] * rz;

  Matx22d Jd;
  _distortion.distort(point_out, Jd);  // distort and Jacobian wrt. keypoint

  // Jacobian including distortion
  jacobian_out(0, 0) = _focalLengthU * Jd(0, 0) * rz;
  jacobian_out(0, 1) = _focalLengthU * Jd(0, 1) * rz;
  jacobian_out(0, 2) = -_focalLengthU
      * (p_C_in[0] * Jd(0, 0) + p_C_in[1] * Jd(0, 1)) * rz2;
  jacobian_out(1, 0) = _focalLengthV * Jd(1, 0) * rz;
  jacobian_out(1, 1) = _focalLengthV * Jd(1, 1) * rz;
  jacobian_out(1, 2) = -_focalLengthV
      * (p_C_in[0] * Jd(1, 0) + p_C_in[1] * Jd(1, 1)) * rz2;

  point_out[0] = _focalLengthU * point_out[0] + _imageCenterU;
  point_out[1] = _focalLengthV * point_out[1] + _imageCenterV;

  return isValid(point_out) && p_C_in[2] > 0;
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::homogeneousToKeypoint(
    const HPoint4d& hp_C_in, Point2d& point_out) const {
  if (hp_C_in[3] < 0) {
    Point3d p_C_in(-hp_C_in[0], -hp_C_in[1], -hp_C_in[2]);
    return euclideanToKeypoint(p_C_in, point_out);
  } else {
    Point3d p_C_in(hp_C_in[0], hp_C_in[1], hp_C_in[2]);
    return euclideanToKeypoint(p_C_in, point_out);
  }
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::homogeneousToKeypoint(
    const HPoint4d& hp_C_in, Point2d& point_out, Matx24d& jacobian_out) const {

  Matx23d jacobian_out_23;
  if (hp_C_in[3] < 0) {
    Point3d p_C_in(-hp_C_in[0], -hp_C_in[1], -hp_C_in[2]);
    bool success = euclideanToKeypoint(p_C_in, point_out, jacobian_out_23);
    jacobian_out = Matx24d(-jacobian_out_23(0, 0), -jacobian_out_23(0, 1),
                           -jacobian_out_23(0, 2), 0.0, -jacobian_out_23(1, 0),
                           -jacobian_out_23(1, 1), -jacobian_out_23(1, 2), 0.0);  //J = -J;
    return success;
  } else {
    Point3d p_C_in(hp_C_in[0], hp_C_in[1], hp_C_in[2]);
    bool success = euclideanToKeypoint(p_C_in, point_out, jacobian_out_23);
    jacobian_out = Matx24d(jacobian_out_23(0, 0), jacobian_out_23(0, 1),
                           jacobian_out_23(0, 2), 0.0, jacobian_out_23(1, 0),
                           jacobian_out_23(1, 1), jacobian_out_23(1, 2), 0.0);
    return success;
  }
}

// cam-to-world: return if the inverse projection is valid
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::keypointToEuclidean(
    const Point2d& point_in, Point3d& p_C_out) const {
  Point2d kp = point_in;  // copy to be modifiable
  kp[0] = (kp[0] - _imageCenterU) / _focalLengthU;
  kp[1] = (kp[1] - _imageCenterV) / _focalLengthV;
  _distortion.undistort(kp);  // revert distortion

  // note: this is not normalized
  p_C_out[0] = kp[0];
  p_C_out[1] = kp[1];
  p_C_out[2] = 1;

  return isValid(point_in);
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::keypointToEuclidean(
    const Point2d& point_in, Point3d& p_C_out,
    Matx32d& inverse_jacobian_out) const {
  Point2d kp = point_in;  // copy to be modifiable
  kp[0] = (kp[0] - _imageCenterU) / _focalLengthU;
  kp[1] = (kp[1] - _imageCenterV) / _focalLengthV;

  Matx22d Jd;
  _distortion.undistort(kp, Jd);  // revert distortion

  // note: this is not normalized
  p_C_out[0] = kp[0];
  p_C_out[1] = kp[1];
  p_C_out[2] = 1;

  inverse_jacobian_out = Matx32d::zeros();

  inverse_jacobian_out(0, 0) = _recip_fu;
  inverse_jacobian_out(1, 1) = _recip_fv;

  inverse_jacobian_out = inverse_jacobian_out * Jd;

  return isValid(point_in);
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::keypointToHomogeneous(
    const Point2d& point_in, HPoint4d& hp_C_out) const {

  // call Euclidean version
  Point3d p_C_out;
  bool success = keypointToEuclidean(point_in, p_C_out);

  // convert size
  hp_C_out = HPoint4d(p_C_out[0], p_C_out[1], p_C_out[2], 0.0);

  return success;
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::keypointToHomogeneous(
    const Point2d& point_in, HPoint4d& hp_C_out,
    Matx42d& inverse_jacobian_out) const {

  // call Euclidean version
  Point3d p_C_out;
  Matx32d inverse_jacobian_out_32;
  bool success = keypointToEuclidean(point_in, p_C_out,
                                     inverse_jacobian_out_32);

  // convert size
  hp_C_out = HPoint4d(p_C_out[0], p_C_out[1], p_C_out[2], 0.0);
  inverse_jacobian_out = Matx42d(inverse_jacobian_out_32(0, 0),
                                 inverse_jacobian_out_32(0, 1),
                                 inverse_jacobian_out_32(1, 0),
                                 inverse_jacobian_out_32(1, 1),
                                 inverse_jacobian_out_32(2, 0),
                                 inverse_jacobian_out_32(2, 1), 0.0, 0.0);

  return success;
}
template<class DISTORTION_T>
bool PinholeCameraGeometry<DISTORTION_T>::isValid(
    const Point2d& point_in) const {
  return point_in(0) > 0.0 && point_in(1) >= 0.0 && point_in(0) <= _pixelsU
      && point_in(1) <= _pixelsV;
}

// get the width and height of the image as supported
template<class DISTORTION_T>
size_t PinholeCameraGeometry<DISTORTION_T>::width() const {
  return _pixelsU;
}
template<class DISTORTION_T>
size_t PinholeCameraGeometry<DISTORTION_T>::height() const {
  return _pixelsV;
}

}
}
