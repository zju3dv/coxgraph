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

class FourDofPriorAutoDiff {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /// \brief Constructor.
  /// @param yaw The yaw prior value.
  /// @param p The translation prior value.
  /// @param covariance The prior covariance.
  FourDofPriorAutoDiff(
      const double yaw,
      const Eigen::Vector3d& p,
      const Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
          defs::pose::kResidualSizeYaw>& covariance) :
    yaw_(yaw),
    p_(p){
    // Compute the square root information from covariance.
    Eigen::LLT<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
          defs::pose::kResidualSizeYaw>> llt(covariance.inverse());
    sqrt_information_ = llt.matrixL();
  }

  template <typename T>
  T NormalizeYaw(const T& angle) const {
    if (angle > T(M_PI))
      return angle - T(2.0 * M_PI);
    else if (angle < T(-M_PI))
    	return angle + T(2.0 * M_PI);
    else
    	return angle;
  }
  
  template <typename T>
  T LogMap(const Eigen::Quaternion<T>& q) const  {
    return T(ceres::atan2(static_cast<T>(2.0) * (q.w() * q.z() + q.x() * q.y()),
        static_cast<T>(1.0) - static_cast<T>(2.0) * (q.y() * q.y() + q.z() * q.z())));
  }

  template <typename T>
  T LogMap(const Eigen::Map<const Eigen::Quaternion<T>>& q) const  {
    return T(ceres::atan2(static_cast<T>(2.0) * (q.w() * q.z() + q.x() * q.y()),
        static_cast<T>(1.0) - static_cast<T>(2.0) * (q.y() * q.y() + q.z() * q.z())));
  }

  // Input Variables:
  // x: The transformation.
  template <typename T>
  bool operator()(const T* const x, T* residuals) const {
    // Unpack parameter blocks.
    const Eigen::Map<const Eigen::Quaternion<T>> q(x);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(x +
        defs::pose::kOrientationBlockSize);
    const T yaw = LogMap(q);

    // Compute the residual
    Eigen::Map<Eigen::Matrix<T, defs::pose::kResidualSizeYaw, 1>>
        residual(residuals);
    residual(0) =  NormalizeYaw(yaw - static_cast<T>(yaw_));
    residual.tail(3) = p - p_.template cast<T>();
    residual = sqrt_information_.cast<T>() * residual;
    return true;
  }

private:
  double yaw_;
  Eigen::Vector3d p_;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
      defs::pose::kResidualSizeYaw> sqrt_information_;
};

} // namespace posegraph

} // namespace robopt

