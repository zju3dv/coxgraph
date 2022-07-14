/*
 * DistortionBase.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#ifndef DISTORTIONBASE_HPP_
#define DISTORTIONBASE_HPP_

#include <brisk/cameras/camera-geometry-base.h>

namespace brisk {
namespace cameras {

// a simple interface for lens distortion applied to normalized (!) image coordinates.
class DistortionBase {
 public:

  virtual ~DistortionBase() {
  }

  // distort an undistorted point
  virtual void distort(Point2d& point) const = 0;
  virtual void distort(Point2d& point, Matx22d& jacobian_out) const = 0;

  // undistort a distorted point
  virtual void undistort(Point2d& point) const = 0;
  virtual void undistort(Point2d& point,
                         Matx22d& inverse_jacobian_out) const = 0;

};

}  // namespace cameras
}  // namespace brisk

#endif /* DISTORTIONBASE_HPP_ */
