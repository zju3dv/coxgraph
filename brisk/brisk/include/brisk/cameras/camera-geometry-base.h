/*
 * CameraGeometryBase.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#ifndef CAMERAGEOMETRYBASE_HPP_
#define CAMERAGEOMETRYBASE_HPP_

#include <opencv2/core/core.hpp>

namespace brisk {
namespace cameras {

typedef cv::Vec2d Point2d;
typedef cv::Vec3d Point3d;
typedef cv::Vec2d Vec2d;
typedef cv::Vec3d Vec3d;
typedef cv::Vec4d HPoint4d;
typedef cv::Matx<double, 2, 2> Matx22d;
typedef cv::Matx<double, 2, 3> Matx23d;
typedef cv::Matx<double, 2, 4> Matx24d;
typedef cv::Matx<double, 3, 2> Matx32d;
typedef cv::Matx<double, 4, 2> Matx42d;

// a simple interface for camera gemetries
class CameraGeometryBase {

 public:
  virtual ~CameraGeometryBase() {
  }

  // world-to-cam: return if the projection is valid and inside the image
  virtual bool euclideanToKeypoint(const Point3d& p_C_in,
                                   Point2d& point_out) const = 0;
  virtual bool euclideanToKeypoint(const Point3d& p_C_in, Point2d& point_out,
                                   Matx23d& jacobian_out) const = 0;
  virtual bool homogeneousToKeypoint(const HPoint4d& hp_C_in,
                                     Point2d& point_out) const = 0;
  virtual bool homogeneousToKeypoint(const HPoint4d& hp_C_in,
                                     Point2d& point_out,
                                     Matx24d& jacobian_out) const = 0;

  // cam-to-world: return if the inverse projection is valid
  virtual bool keypointToEuclidean(const Point2d& point_in,
                                   Point3d& p_C_out) const = 0;
  virtual bool keypointToEuclidean(const Point2d& point_in, Point3d& p_C_out,
                                   Matx32d& inverse_jacobian_out) const = 0;
  virtual bool keypointToHomogeneous(const Point2d& point_in,
                                     HPoint4d& hp_C_out) const = 0;
  virtual bool keypointToHomogeneous(const Point2d& point_in,
                                     HPoint4d& hp_C_out,
                                     Matx42d& inverse_jacobian_out) const = 0;

  // check boundaries
  virtual bool isValid(const Point2d& point_in) const = 0;

  // get the width and height of the image as supported
  virtual size_t width() const = 0;
  virtual size_t height() const = 0;
};

}  // namespace cameras
}  // namespace brisk

#endif /* CAMERAGEOMETRYBASE_HPP_ */
