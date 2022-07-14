/*
 * EquidistantDistortion.hpp
 *
 *  Created on: Dec 26, 2013
 *      Author: lestefan
 */

#ifndef EQUIDISTANTDISTORTION_HPP_
#define EQUIDISTANTDISTORTION_HPP_

#include <brisk/cameras/distortion-base.h>

namespace brisk {
namespace cameras {

//EquidistantDistortion lens distortion applied to normalized (!) image coordinates.
class EquidistantDistortion : public DistortionBase {
 public:

  virtual ~EquidistantDistortion() {
  }

  // constructor that initializes all values.
  inline EquidistantDistortion(double k1, double k2, double k3, double k4);

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
  void setK3(double k3) {
    _k3 = k3;
  }
  void setK4(double k4) {
    _k4 = k4;
  }

  // getters
  double k1() {
    return _k1;
  }
  double k2() {
    return _k2;
  }
  double k3() {
    return _k3;
  }
  double k4() {
    return _k4;
  }

 private:

  double _k1;
  double _k2;
  double _k3;
  double _k4;

};

}  // namespace cameras
}  // namespace brisk

#include "implementation/equidistant-distortion.h"

#endif /* EQUIDISTANTDISTORTION_HPP_ */
