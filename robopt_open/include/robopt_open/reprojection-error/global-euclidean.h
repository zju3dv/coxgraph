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
 * global-euclidean.h
 * @brief Header file for reprojection residuals of landmarks to cameras 
 *        expressed in global, euclidean coordinates.
 * @author: Marco Karrer
 * Created on: Mar 19, 2018
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

template <typename CameraType, typename DistortionType>
class GlobalEuclideanReprError
    : public ceres::SizedCostFunction<
        defs::visual::kResidualSize, defs::visual::kPoseBlockSize, 
        defs::visual::kPoseBlockSize, defs::visual::kPositionBlockSize,
        CameraType::parameterCount(),
        DistortionType::parameterCount()> {

public:
  /// \brief Construct a cost function using the reprojection error.
  /// @param measurement The landmark observation in the image (distorted)
  /// @param pixel_sigma The measurement standard deviation (pixels).
  /// @param camera The aslam camera object.
  GlobalEuclideanReprError(
      const Eigen::Vector2d& measurement, double pixel_sigma,
      const CameraType* camera) :
          measurement_(measurement),          
          camera_ptr_(camera) {
    pixel_sigma_inverse_ = 1.0/pixel_sigma;
  }

  virtual ~GlobalEuclideanReprError() {};

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks.
  enum {
    kIdxImuPose,
    kIdxCameraToImu,
    kIdxLandmark,
    kIdxCameraIntrinsics,
    kIdxCameraDistortion
  };

  // The representation for Jacobians computed by this object.
  typedef Eigen::Matrix<double, defs::visual::kResidualSize, 
                        defs::visual::kPoseBlockSize, Eigen::RowMajor>
      PoseJacobian;

  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPositionBlockSize, Eigen::RowMajor>
      PositionJacobian;

  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        CameraType::parameterCount(), Eigen::RowMajor>
      IntrinsicsJacobian;

  typedef Eigen::Matrix<double, defs::visual::kResidualSize, Eigen::Dynamic,
                        Eigen::RowMajor>
      DistortionJacobian;

  // Store information related to the observation
  Eigen::Vector2d measurement_;
  double pixel_sigma_inverse_;
  const CameraType* camera_ptr_;
}; 


} // namespace reprojection

} // namespace robopt

#include "./global-euclidean-inl.h"
