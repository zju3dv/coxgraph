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
 * smart-projection.h
 * @brief Header file for smart-reprojection factor.
 * @author: Marco Karrer
 * Created on: Nov 6, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>

#include <aslam/cameras/camera.h>

#include <robopt_open/common/common.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/local-parameterization/pose-quaternion-local-param.h>
#include <robopt_open/reprojection-error/triangulation.h>

#include <fstream>
/// \brief robopt Main namespace of this package
namespace robopt {

namespace reprojection {

template<typename CameraType, typename DistortionType>
class SmartProjectionError
    : public ceres::CostFunction {

public:
  /// \brief Construct a cost function using the smart reprojection factors.
  ///        Note: For now there is no support for intrinsics optimization.
  /// @param measurements The landmark observations in the image (distorted).
  /// @param pixel_sigma The measurement standard deviation (pixels).
  ///        Note: for now it is assumed all projections detection have the same
  ///        uncertainty.
  /// @param cameras The aslam camera objects associated to the measurements.
  /// @param rot_relin_threshold Threshold for
  /// @param l_W_init Optional initial guess for the landmark
  SmartProjectionError(
      const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<
        Eigen::Vector2d>>& measurements,
      const double pixel_sigma,
      const std::vector<CameraType*>& cameras) :
    measurements_(measurements),
    camera_ptrs_(cameras),
    num_cams_(measurements.size()) {
    pixel_sigma_inverse_ = 1.0/pixel_sigma;
    set_num_residuals(measurements.size()*2 - defs::visual::kPositionBlockSize);

    // Set parameter sizes in: parameter_block_sizes_
    // Note: For every observation we have 2 blocks ([pose, extrinsics])
    for (size_t i = 0; i < num_cams_; ++i) {
      // Block for the actual pose
      mutable_parameter_block_sizes()->push_back(defs::visual::kPoseBlockSize);

      // Block for the extrinsics transformation
      mutable_parameter_block_sizes()->push_back(defs::visual::kPoseBlockSize);
      Eigen::Vector3d back_proj;
      camera_ptrs_[i]->backProject3(measurements[i], &back_proj);
      meas_normalized_.push_back(back_proj.head<2>()/back_proj[2]);
    }

    // Initialize cache data
    last_l_W_ = new Eigen::Vector3d(0.0, 0.0, 0.0);
    has_landmark_ = new bool;
    (*has_landmark_) = false;
    outliers_ = new std::vector<bool>(num_cams_, false);
  }

  virtual ~SmartProjectionError() {};

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  void setInitialGuess(const Eigen::Vector3d& l_W_init) {
    (*last_l_W_) = l_W_init;
    (*has_landmark_) = true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks
  enum {
    kIdxImuPose,
    kIdxCameraToImu,
    kIdxCameraIntrinsics,
    kIdxCameraDistortion
  };

  // Typedef for storing the mappings
  typedef std::vector<Eigen::Map<const Eigen::Quaterniond>,
      Eigen::aligned_allocator<Eigen::Map<const Eigen::Quaterniond>>>
  VectorConstMapQuaternion;
  typedef std::vector<Eigen::Map<const Eigen::Vector3d>,
      Eigen::aligned_allocator<Eigen::Map<const Eigen::Vector3d>>>
  VectorConstMapVector3;
  typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<
      Eigen::Vector2d>> VectorVector2;
  typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<
      Eigen::Matrix4d>> VectorMatrix4;

  // The representation for Jacobians computed by this object.
  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPoseBlockSize, Eigen::RowMajor>
      PoseJacobian;

  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPositionBlockSize, Eigen::RowMajor>
      PositionJacobian;
  typedef Eigen::Matrix<double, defs::visual::kResidualSize, 6>
      PoseJacobianMinimal;
  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPositionBlockSize>
      VisualJacobianType;

  // Store the information related to the observation
  VectorVector2 measurements_;
  VectorVector2 meas_normalized_;

  const size_t num_cams_;
  double pixel_sigma_inverse_;
  std::vector<CameraType*> camera_ptrs_;

  // Cache data to possibly reduce complexity for subsequent iterations
  // Last triangulated point
  Eigen::Vector3d* last_l_W_;
  bool* has_landmark_;
  std::vector<bool>* outliers_;

};

} // namespace reprojection

} // namespace robopt

#include "./smart-projection-inl.h"
