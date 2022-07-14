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
 * four-dof-between.h
 * @brief Header file for relative pose constraints between two frames.
 *        (constraints only positions and yaw angle).
 * @author: Marco Karrer
 * Created on: Aug 18, 2018
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

class FourDofBetweenError
    : public ceres::SizedCostFunction<
        defs::pose::kResidualSizeYaw, defs::pose::kPoseBlockSize,
        defs::pose::kPoseBlockSize, defs::pose::kPoseBlockSize,
        defs::pose::kPoseBlockSize> {

public:
  /// \brief Construct a cost function using the relative pose error (4DoF).
  /// @param rotation_measurement Rotation from frame 2 into frame 1 (yaw angle)
  /// @param translation_measurement Translation from frame 2 to frame 1.
  /// @param sqrt_information Square-Root Information matrix of the relative
  ///         measurement (ordering: [rotation, translation])
  FourDofBetweenError(
      const double rotation_measurement, 
      const Eigen::Vector3d& translation_measurement, 
      const Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
                          defs::pose::kResidualSizeYaw>& sqrt_information,
      defs::pose::PoseErrorType error_term_type) :
          rotation_measurement_(rotation_measurement),          
          translation_measurement_(translation_measurement),
          sqrt_information_(sqrt_information),
          error_term_type_(error_term_type) {
  }

  virtual ~FourDofBetweenError() {};

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
  typedef Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
                        defs::pose::kPoseBlockSize, Eigen::RowMajor>
      PoseJacobian;
  typedef Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
                        6, Eigen::RowMajor>
      PoseJacobianMin;
  typedef Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
                        4, Eigen::RowMajor>
      PoseYawJacobianMin;

  // Store information related to the observation
  double rotation_measurement_;
  Eigen::Vector3d translation_measurement_;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 
      defs::pose::kResidualSizeYaw> sqrt_information_;
  defs::pose::PoseErrorType error_term_type_;
}; 

class FourDofBetweenErrorAutoDiff {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FourDofBetweenErrorAutoDiff(
      const double rotation_measurement,
      const Eigen::Vector3d& translation_measurement,
      const Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
          defs::pose::kResidualSizeYaw>& sqrt_information,
      defs::pose::PoseErrorType error_term_type) :
    rotation_measurement_(rotation_measurement),
    translation_measurement_(translation_measurement),
    sqrt_information_(sqrt_information),
    error_term_type_(error_term_type) {}

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

  template <typename T>
  bool operator()(const T* const x_W_S1 , const T* const x_W_S2,
      const T* const x_S_C1, const T* const x_S_C2, T* residuals) const {
    // Unpack parameter blocks.
    const Eigen::Map<const Eigen::Quaternion<T>> q_W_S1(x_W_S1);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_W_S1(x_W_S1 +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_W_S2(x_W_S2);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_W_S2(x_W_S2 +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_S_C1(x_S_C1);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_S_C1(x_S_C1 +
        defs::pose::kOrientationBlockSize);
    Eigen::Map<const Eigen::Quaternion<T>> q_S_C2(x_S_C2);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_S_C2(x_S_C2 +
        defs::pose::kOrientationBlockSize);

    // Compute the residual
    Eigen::Map<Eigen::Matrix<T, defs::pose::kResidualSizeYaw, 1>>
        residual(residuals);

    if (error_term_type_ == defs::pose::PoseErrorType::kVisual) {
      const Eigen::Quaternion<T> q_W_C1 = q_W_S1 * q_S_C1;
      const T yaw_C1 = LogMap(q_W_C1);
      Eigen::Matrix<T, 3, 1> p_W_C1 = q_W_S1 * p_S_C1 + p_W_S1;
      Eigen::Quaternion<T> q_W_C2 = q_W_S2 * q_S_C2;
      Eigen::Matrix<T, 3, 1> p_W_C2 = q_W_S2 * p_S_C2 + p_W_S2;
      const T yaw_C2 = LogMap(q_W_C2);
      Eigen::Matrix<T, 3, 1> p_C2_C1 = q_W_C2.inverse() * (p_W_C1 - p_W_C2);

      residual(0) = NormalizeYaw(yaw_C1 - yaw_C2 -
          static_cast<T>(rotation_measurement_));
      residual.tail(3) = p_C2_C1 - translation_measurement_.template cast<T>();
      residual = sqrt_information_.cast<T>() * residual;
    } else if (error_term_type_ == defs::pose::PoseErrorType::kImu) {
      const T yaw_S1 = LogMap(q_W_S1);
      const T yaw_S2 = LogMap(q_W_S2);
      Eigen::Matrix<T, 3, 1> p_S2_S1 = q_W_S2.inverse() * (p_W_S1 - p_W_S2);

      residual(0) = NormalizeYaw(yaw_S1 - yaw_S2 -
           static_cast<T>(rotation_measurement_));
      residual.tail(3) = p_S2_S1 - translation_measurement_.template cast<T>();
      residual = sqrt_information_.cast<T>() * residual;
    } else {
      LOG(FATAL) << "Unknown pose error term type.";
      return false;
    }

    return true;
  }

private:
  double rotation_measurement_;
  Eigen::Vector3d translation_measurement_;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
      defs::pose::kResidualSizeYaw> sqrt_information_;
  defs::pose::PoseErrorType error_term_type_;
};

class FourDofBetweenErrorAutoDiff2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FourDofBetweenErrorAutoDiff2(
      const double rotation_measurement,
      const Eigen::Vector3d& translation_measurement,
      const Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
          defs::pose::kResidualSizeYaw>& sqrt_information,
      defs::pose::PoseErrorType error_term_type) :
    rotation_measurement_(rotation_measurement),
    translation_measurement_(translation_measurement),
    sqrt_information_(sqrt_information),
    error_term_type_(error_term_type) {}

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

  template <typename T>
  bool operator()(const T* const x_W_M1, const T* const x_W_M2,
      const T* const x_M_S1 , const T* const x_M_S2, 
      const T* const x_S_C1, const T* const x_S_C2, T* residuals) const {
    // Unpack parameter blocks.
    const Eigen::Map<const Eigen::Quaternion<T>> q_W_M1(x_W_M1);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_W_M1(x_W_M1 +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_W_M2(x_W_M2);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_W_M2(x_W_M2 +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_M_S1(x_M_S1);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_M_S1(x_M_S1 +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_M_S2(x_M_S2);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_M_S2(x_M_S2 +
        defs::pose::kOrientationBlockSize);
    const Eigen::Map<const Eigen::Quaternion<T>> q_S_C1(x_S_C1);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_S_C1(x_S_C1 +
        defs::pose::kOrientationBlockSize);
    Eigen::Map<const Eigen::Quaternion<T>> q_S_C2(x_S_C2);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_S_C2(x_S_C2 +
        defs::pose::kOrientationBlockSize);

    // Compute the residual
    Eigen::Map<Eigen::Matrix<T, defs::pose::kResidualSizeYaw, 1>>
        residual(residuals);

    if (error_term_type_ == defs::pose::PoseErrorType::kVisual) {
      const Eigen::Quaternion<T> q_W_C1 = q_W_M1 * q_M_S1 * q_S_C1;
      const T yaw_C1 = LogMap(q_W_C1);
      Eigen::Matrix<T, 3, 1> p_W_C1 = q_W_M1 * (q_M_S1 * p_S_C1 + 
          p_M_S1) + p_W_M1;
      Eigen::Quaternion<T> q_W_C2 = q_W_M2 * q_M_S2 * q_S_C2;
      Eigen::Matrix<T, 3, 1> p_W_C2 = q_W_M2 * (q_M_S2 * p_S_C2 + 
          p_M_S2) + p_W_M2;
      const T yaw_C2 = LogMap(q_W_C2);
      Eigen::Matrix<T, 3, 1> p_C2_C1 = q_W_C2.inverse() * (p_W_C1 - p_W_C2);

      residual(0) = NormalizeYaw(yaw_C1 - yaw_C2 -
          static_cast<T>(rotation_measurement_));
      residual.tail(3) = p_C2_C1 - translation_measurement_.template cast<T>();
      residual = sqrt_information_.cast<T>() * residual;
    } else if (error_term_type_ == defs::pose::PoseErrorType::kImu) {
      const Eigen::Quaternion<T> q_W_S1 = q_W_M1 * q_M_S1;
      const Eigen::Quaternion<T> q_W_S2 = q_W_M2 * q_M_S2;
      const Eigen::Matrix<T, 3, 1> p_W_S1 = q_W_M1 * p_M_S1 + p_W_M1;
      const Eigen::Matrix<T, 3, 1> p_W_S2 = q_W_M2 * p_M_S2 + p_W_M2;
      const T yaw_S1 = LogMap(q_W_S1);
      const T yaw_S2 = LogMap(q_W_S2);
      Eigen::Matrix<T, 3, 1> p_S2_S1 = q_W_S2.inverse() * (p_W_S1 - p_W_S2);

      residual(0) = NormalizeYaw(yaw_S1 - yaw_S2 -
           static_cast<T>(rotation_measurement_));
      residual.tail(3) = p_S2_S1 - translation_measurement_.template cast<T>();
      residual = sqrt_information_.cast<T>() * residual;
    } else {
      LOG(FATAL) << "Unknown pose error term type.";
      return false;
    }

    return true;
  }

private:
  double rotation_measurement_;
  Eigen::Vector3d translation_measurement_;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
      defs::pose::kResidualSizeYaw> sqrt_information_;
  defs::pose::PoseErrorType error_term_type_;
};

} // namespace posegraph

} // namespace robopt
