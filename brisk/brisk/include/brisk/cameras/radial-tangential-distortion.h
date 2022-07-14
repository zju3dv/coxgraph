/*
 * RadialTangentialDistortion.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#ifndef RADIALTANGENTIALDISTORTION_HPP_
#define RADIALTANGENTIALDISTORTION_HPP_

#include <brisk/cameras/distortion-base.h>

namespace brisk {
namespace cameras {

//RadialTangentialDistortion lens distortion applied to normalized (!) image coordinates.
class RadialTangentialDistortion : public DistortionBase {
 public:

  virtual ~RadialTangentialDistortion() {
  }
  ;

  // constructor that initializes all values.
  inline RadialTangentialDistortion(double k1, double k2, double p1, double p2);

  // inherited interface

  // distort an undistorted point
  virtual inline void distort(Point2d& point) const;
  virtual inline void distort(Point2d& point, Matx22d& jacobian_out) const;

  // undistort a distorted point
  virtual inline void undistort(Point2d& point) const;
  virtual inline void undistort(Point2d& point,
                                Matx22d& inverse_jacobian_out) const;

  // setters
  void setK1(double k1) {
    _k1 = k1;
  }
  void setK2(double k2) {
    _k2 = k2;
  }
  void setP1(double p1) {
    _p1 = p1;
  }
  void setP2(double p2) {
    _p2 = p2;
  }

  // getters
  // the first radial distortion parameter
  double k1() {
    return _k1;
  }
  // the second radial distortion parameter
  double k2() {
    return _k2;
  }
  // the first tangential distortion parameter
  double p1() {
    return _p1;
  }
  // the second tangential distortion parameter
  double p2() {
    return _p2;
  }

 private:
  // the first radial distortion parameter
  double _k1;
  // the second radial distortion parameter
  double _k2;
  // the first tangential distortion parameter
  double _p1;
  // the second tangential distortion parameter
  double _p2;

};

}  // namespace cameras
}  // namespace brisk

#include "implementation/radial-tangential-distortion.h"

#endif /* RADIALTANGENTIALDISTORTION_HPP_ */
