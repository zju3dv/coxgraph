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
 * smart-relative-distance.h
 * @brief Header file for relative-distance constraints.
 * @author: Marco Karrer
 * Created on: Nov 26, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>

#include <robopt_open/common/common.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/local-parameterization/pose-quaternion-local-param.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace posegraph {

class RelativeDistanceError
    : public ceres::SizedCostFunction<
        1, defs::pose::kPoseBlockSize, defs::pose::kPoseBlockSize, 
        defs::pose::kPositionBlockSize, defs::pose::kPositionBlockSize> {

public:
  /// \brief Construct a cost function using a scalar relative distance.
  /// @param distance_measurement The measured distance between the two frames.
  /// @param sqrt_inforamtion Square-root information (simply a scalar) for
  ///         the distance measurement.
  RelativeDistanceError(
      const double distance_measurement,
      const double sqrt_information) :
      measurement_(distance_measurement),
      sqrt_information_(sqrt_information) {

  }

  virtual ~RelativeDistanceError() {};

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

  // Store informaiton related to the observation
  double measurement_;
  double sqrt_information_;
}; 


class RelativeDistanceFixedError
    : public ceres::SizedCostFunction<
        1, defs::pose::kPoseBlockSize, defs::pose::kPositionBlockSize> {

public:
  /// \brief Construct a cost function using a scalar relative distance.
  ///        In this version the pose& extrinsics of the second frame are fixed.
  /// @param distance_measurement The measured distance between the two frames.
  /// @param sqrt_inforamtion Square-root information (simply a scalar) for
  ///         the distance measurement.
  /// @param q_W_S2 The rotation of the second frame.
  /// @param p_W_S2 The translation of the second frame.
  /// @param p_S_U2 The extrinsics of the second frame.
  RelativeDistanceFixedError(
      const double distance_measurement,
      const double sqrt_information,
      const Eigen::Quaterniond& q_W_S2,
      const Eigen::Vector3d& p_W_S2,
      const Eigen::Vector3d& p_S_U2) :
    measurement_(distance_measurement),
    sqrt_information_(sqrt_information),
    q_W_S2_(q_W_S2),
    p_W_S2_(p_W_S2),
    p_S_U2_(p_S_U2) {

  }

  virtual ~RelativeDistanceFixedError() {};

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks.
  enum {
    kIdxPose1,
    kIdxExtrinsics1,
  };

  // Store informaiton related to the observation
  const double measurement_;
  const double sqrt_information_;
  const Eigen::Quaterniond q_W_S2_;
  const Eigen::Vector3d p_W_S2_;
  const Eigen::Vector3d p_S_U2_;
};

} // namespace posegraph

} // namespace robopt
