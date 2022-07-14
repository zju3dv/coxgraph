/*
* Copyright (c) 2018, Vision for Robotics Lab
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the Vision for Robotics Lab, ETH Zurich nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

/*
 * relative-euclidean.h
 * @brief Header file for reprojection residuals of fixed landmarks expressed in
 *    a relative frame to a camera.
 * @author: Marco Karrer
 * Created on: Aug 17, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/sized_cost_function.h>

#include <aslam/cameras/camera.h>

#include <robopt_open/common/common.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/local-parameterization/quaternion-local-param.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace reprojection {

// Only optimize the relative pose between the two CF. Does not make sense to
// allow extrinsics and intrinsics optimization.
template <typename CameraType, typename DistortionType>
class RelativeEuclideanReprError 
    : public ceres::SizedCostFunction<
        defs::visual::kResidualSize, defs::visual::kPoseBlockSize> {

public:
  /// \brief Construct a cost function using the reprojection error.
  /// @param measurement The landmark observation in the image (distorted).
  /// @param pixel_sigma the measurement standard deviation (pixels).
  /// @param camera The aslam camera object.
  /// @param point_ref The point in the reference frame.
  /// @param error_term_type The type of error (i.e. in which direction the
  ///         transformation is performed).
  RelativeEuclideanReprError(
      const Eigen::Vector2d& measurement,
      const double pixel_sigma,
      const CameraType* camera,
      const Eigen::Vector3d& point_ref,
      const defs::visual::RelativeProjectionType error_term_type) :
          measurement_(measurement),
          camera_ptr_(camera),
          point_ref_(point_ref),
          error_term_type_(error_term_type) {
    pixel_sigma_inverse_ = 1.0/pixel_sigma;
  }

  virtual ~RelativeEuclideanReprError() {};

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The representation for Jacobians computed by this object.
  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPoseBlockSize, Eigen::RowMajor>
      PoseJacobian;

  // Store information related to the observation
  Eigen::Vector2d measurement_;
  Eigen::Vector3d point_ref_;
  double pixel_sigma_inverse_;
  const CameraType* camera_ptr_;
  defs::visual::RelativeProjectionType error_term_type_;

};

} // namespace reprojection

} // namespace robopt

#include "./relative-euclidean-inl.h"
