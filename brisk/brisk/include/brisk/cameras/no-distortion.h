/*
 * NoDistortion.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#ifndef NODISTORTION_HPP_
#define NODISTORTION_HPP_

#include <brisk/cameras/distortion-base.h>

namespace brisk {
namespace cameras {

//no lens distortion applied to normalized (!) image coordinates.
class NoDistortion : public DistortionBase {
 public:

  virtual ~NoDistortion() {
  }

  // distort an undistorted point
  virtual void distort(Point2d& /*point*/) const {}

  virtual void distort(Point2d& /*point*/, Matx22d& jacobian_out) const {
    jacobian_out = Matx22d::eye();
  }

  // undistort a distorted point
  virtual void undistort(Point2d& /*point*/) const {}

  virtual void undistort(Point2d& /*point*/, Matx22d& inverse_jacobian_out) const {
    inverse_jacobian_out = Matx22d::eye();
  }

};

}  // namespace cameras
}  // namespace brisk

#endif /* NODISTORTION_HPP_ */
