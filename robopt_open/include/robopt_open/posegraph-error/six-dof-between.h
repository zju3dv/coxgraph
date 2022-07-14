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
 * six-dof-between.h
 * @brief Header file for relative pose constraints between two frames.
 * @author: Marco Karrer
 * Created on: Mar 22, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/sized_cost_function.h>

#include <robopt_open/common/common.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/local-parameterization/quaternion-local-param.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace posegraph {

class SixDofBetweenError
    : public ceres::SizedCostFunction<
        defs::pose::kResidualSize, defs::pose::kPoseBlockSize,
        defs::pose::kPoseBlockSize, defs::pose::kPoseBlockSize,
        defs::pose::kPoseBlockSize> {

public:
  /// \brief Construct a cost function using the relative pose error (6DoF).
  /// @param rotation_measurement Rotation from frame 2 into frame 1
  /// @param translation_measurement Translation from frame 2 to frame 1.
  /// @param sqrt_information Square-Root Information matrix of the relative
  ///         measurement (ordering: [rotation, translation])
  SixDofBetweenError(
      const Eigen::Quaterniond& rotation_measurement, 
      const Eigen::Vector3d& translation_measurement, 
      const Eigen::Matrix<double, defs::pose::kResidualSize,
                          defs::pose::kResidualSize>& sqrt_information,
      defs::pose::PoseErrorType error_term_type) :
          rotation_measurement_(rotation_measurement),          
          translation_measurement_(translation_measurement),
          sqrt_information_(sqrt_information),
          error_term_type_(error_term_type) {
  }

  virtual ~SixDofBetweenError() {};

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks.
  enum {
    kIdxPose1,
    kIdxPose2,
    kIdxExtrinsics1,
    kIdxExtrinsics2
  };

  // The representation for Jacobians computed by this object.
  typedef Eigen::Matrix<double, defs::pose::kResidualSize,
                        defs::pose::kPoseBlockSize, Eigen::RowMajor>
      PoseJacobian;
  typedef Eigen::Matrix<double, defs::pose::kResidualSize,
                        6, Eigen::RowMajor>
      PoseJacobianMin;

  // Store information related to the observation
  Eigen::Quaterniond rotation_measurement_;
  Eigen::Vector3d translation_measurement_;
  Eigen::Matrix<double, defs::pose::kResidualSize, defs::pose::kResidualSize>
      sqrt_information_;
  defs::pose::PoseErrorType error_term_type_;
}; 


} // namespace reprojection

} // namespace robopt
