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
 * gps-error-autodiff.h
 * @brief Header file for GPS error terms (auto-diff implementation).
 * @author: Marco Karrer
 * Created on: Aug 25, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/sized_cost_function.h>

#include <robopt_open/common/common.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/local-parameterization/quaternion-local-param.h>
#include <robopt_open/local-parameterization/pose-quaternion-yaw-local-param.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace posegraph {

class GpsErrorAutoDiff {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /// \brief Constructor.
  /// @param gps_measurement The gps measurement in the GPS reference frame.
  /// @param q_rel, p_rel The relative transformation from the frame where
  ///     the measurement was recorded and the (key)frame whose pose should
  ///     be optimized.
  /// @param covariance The gps measurement covariance.
  GpsErrorAutoDiff(
      const Eigen::Vector3d& gps_measurement,
      const Eigen::Quaterniond& q_rel,
      const Eigen::Vector3d& p_rel,
      const Eigen::Matrix<double, defs::pose::kResidualSizePosition,
          defs::pose::kResidualSizePosition>& covariance) :
    gps_measurement_(gps_measurement),
    q_rel_(q_rel),
    p_rel_(p_rel){
    // Compute the square root information from covariance.
    Eigen::LLT<Eigen::Matrix<double, defs::pose::kResidualSizePosition,
          defs::pose::kResidualSizePosition>> llt(covariance.inverse());
    sqrt_information_ = llt.matrixL();
  }

  // Input Variables:
  // x_R_W: Transformation from map origin to the GPS reference frame
  // x_W_S: The keyframe/frame pose.
  // x_S_B: The translation of the GPS antenna in the IMU (S) frame.
  template <typename T>
  bool operator()(const T* const x_R_W , const T* const x_W_S,
      const T* const x_S_B, T* residuals) const {
    // Unpack parameter blocks.
    const Eigen::Map<const Eigen::Quaternion<T>> q_R_W(x_R_W);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_R_W(x_R_W +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_W_S(x_W_S);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_W_S(x_W_S +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_S_B(x_S_B);

    // First transform the imu frame to the relative odometry frame
    const Eigen::Quaternion<T> q_W_Sm = q_W_S * (q_rel_.template cast<T>());
    const Eigen::Matrix<T, 3, 1> p_W_Sm = q_W_S * p_rel_.template cast<T>() +
        p_W_S;
    const Eigen::Matrix<T, 3, 1> p_W_B = q_W_Sm * p_S_B + p_W_Sm;
    const Eigen::Matrix<T, 3, 1> p_R_B = q_R_W * p_W_B + p_R_W;

    // Compute the residual
    Eigen::Map<Eigen::Matrix<T, defs::pose::kResidualSizePosition, 1>>
        residual(residuals);
    residual = p_R_B - gps_measurement_.template cast<T>();
    residual = sqrt_information_.cast<T>() * residual;

    return true;
  }

private:
  Eigen::Vector3d gps_measurement_;
  Eigen::Quaterniond q_rel_;
  Eigen::Vector3d p_rel_;
  Eigen::Matrix<double, defs::pose::kResidualSizePosition,
      defs::pose::kResidualSizePosition> sqrt_information_;
};

} // namespace posegraph

} // namespace robopt

