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
 * relative-distance.cpp
 * @brief Implementation file for relative distance constraints between two frames.
 * @author: Marco Karrer
 * Created on: Nov 28, 2018
 */

#include <posegraph-error/relative-distance.h>

namespace robopt {

namespace posegraph {

bool RelativeDistanceError::Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const {
  // Coordinate frames:
  //  W = global
  //  S = IMU frame, expressed in W
  //  U = Measurement frame, expressed in S
  //  1 = index of first reference
  //  2 = index of second reference

  // Unpack parameter blocks.
  Eigen::Map<const Eigen::Quaterniond> q_W_S1(parameters[kIdxPose1]);
  Eigen::Map<const Eigen::Vector3d> p_W_S1(parameters[kIdxPose1] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Quaterniond> q_W_S2(parameters[kIdxPose2]);
  Eigen::Map<const Eigen::Vector3d> p_W_S2(parameters[kIdxPose2] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Vector3d> p_S_U1(parameters[kIdxExtrinsics1]);
  Eigen::Map<const Eigen::Vector3d> p_S_U2(parameters[kIdxExtrinsics2]);

  // Compute the residual
  const Eigen::Vector3d p_W_U1 = q_W_S1 * p_S_U1 + p_W_S1;
  const Eigen::Vector3d p_W_U2 = q_W_S2 * p_S_U2 + p_W_S2;
  const Eigen::Vector3d difference = p_W_U1 - p_W_U2;
  const double distance = difference.norm();

  (*residuals) = (distance - measurement_) * sqrt_information_;

  // Compute the jacobians
  if (jacobians) {
    // Norm jacobian is always needed, so only compute it once.
    Eigen::Matrix<double, 1, 3> J_res_wrt_difference;
    if (distance > 1e-8) {
      J_res_wrt_difference(0, 0) = difference[0]/distance;
      J_res_wrt_difference(0, 1) = difference[1]/distance;
      J_res_wrt_difference(0, 2) = difference[2]/distance;
    } else {
      J_res_wrt_difference(0, 0) = std::copysign(1.0, difference[0]);
      J_res_wrt_difference(0, 1) = std::copysign(1.0, difference[1]);;
      J_res_wrt_difference(0, 2) = std::copysign(1.0, difference[2]);;
    }
    local_param::QuaternionLocalParameterization quat_parameterization;

    if (jacobians[kIdxPose1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>
          J(jacobians[kIdxPose1]);
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parameterization.ComputeJacobian(
            q_W_S1.coeffs().data(), J_quat_local_param.data());
      Eigen::Matrix<double, 6, defs::pose::kPoseBlockSize> J_local;
      J_local.setZero();
      J_local.block<3,4>(0,0) = J_quat_local_param.transpose() * 4.0;
      J_local.block<3,3>(3,4) = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 3, 6> J_p_W_U1_wrt_T_W_S1;
      J_p_W_U1_wrt_T_W_S1.setZero();
      J_p_W_U1_wrt_T_W_S1.block<3, 3>(0, 0) = -q_W_S1.toRotationMatrix() *
          common::skew(p_S_U1);
      J_p_W_U1_wrt_T_W_S1.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
      J = sqrt_information_ * J_res_wrt_difference *
          J_p_W_U1_wrt_T_W_S1 * J_local;
    }

    if (jacobians[kIdxPose2]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>
          J(jacobians[kIdxPose2]);
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parameterization.ComputeJacobian(
            q_W_S2.coeffs().data(), J_quat_local_param.data());
      Eigen::Matrix<double, 6, defs::pose::kPoseBlockSize> J_local;
      J_local.setZero();
      J_local.block<3,4>(0,0) = J_quat_local_param.transpose() * 4.0;
      J_local.block<3,3>(3,4) = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 3, 6> J_p_W_U2_wrt_T_W_S2;
      J_p_W_U2_wrt_T_W_S2.setZero();
      J_p_W_U2_wrt_T_W_S2.block<3, 3>(0, 0) = -q_W_S2.toRotationMatrix() *
          common::skew(p_S_U2);
      J_p_W_U2_wrt_T_W_S2.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
      J = sqrt_information_ * J_res_wrt_difference *
          (-J_p_W_U2_wrt_T_W_S2) * J_local;
    }

    if (jacobians[kIdxExtrinsics1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>
          J(jacobians[kIdxExtrinsics1]);
      Eigen::Matrix3d J_p_W_U1_wrt_p_S_U1 = q_W_S1.toRotationMatrix();
      J = sqrt_information_ * J_res_wrt_difference * J_p_W_U1_wrt_p_S_U1;
    }

    if (jacobians[kIdxExtrinsics2]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>
          J(jacobians[kIdxExtrinsics2]);
      Eigen::Matrix3d J_p_W_U2_wrt_p_S_U2 = q_W_S2.toRotationMatrix();
      J = sqrt_information_ * J_res_wrt_difference * (-J_p_W_U2_wrt_p_S_U2);
    }
  }

  return true;
}

bool RelativeDistanceFixedError::Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const {
  // Coordinate frames:
  //  W = global
  //  S = IMU frame, expressed in W
  //  U = Measurement frame, expressed in S
  //  1 = index of first reference
  //  2 = index of second reference

  // Unpack parameter blocks.
  Eigen::Map<const Eigen::Quaterniond> q_W_S1(parameters[kIdxPose1]);
  Eigen::Map<const Eigen::Vector3d> p_W_S1(parameters[kIdxPose1] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Vector3d> p_S_U1(parameters[kIdxExtrinsics1]);

  // Compute the residual
  const Eigen::Vector3d p_W_U1 = q_W_S1 * p_S_U1 + p_W_S1;
  const Eigen::Vector3d p_W_U2 = q_W_S2_ * p_S_U2_ + p_W_S2_;
  const Eigen::Vector3d difference = p_W_U1 - p_W_U2;
  const double distance = difference  .norm();

  (*residuals) = (distance - measurement_) * sqrt_information_;

  // Compute the jacobians
  if (jacobians) {
    // Norm jacobian is always needed, so only compute it once.
    Eigen::Matrix<double, 1, 3> J_res_wrt_difference;
    J_res_wrt_difference(0, 0) = difference[0]/distance;
    J_res_wrt_difference(0, 1) = difference[1]/distance;
    J_res_wrt_difference(0, 2) = difference[2]/distance;
    local_param::QuaternionLocalParameterization quat_parameterization;

    if (jacobians[kIdxPose1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>
          J(jacobians[kIdxPose1]);
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parameterization.ComputeJacobian(
            q_W_S1.coeffs().data(), J_quat_local_param.data());
      Eigen::Matrix<double, 6, defs::pose::kPoseBlockSize> J_local;
      J_local.setZero();
      J_local.block<3,4>(0,0) = J_quat_local_param.transpose() * 4.0;
      J_local.block<3,3>(3,4) = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 3, 6> J_p_W_U1_wrt_T_W_S1;
      J_p_W_U1_wrt_T_W_S1.setZero();
      J_p_W_U1_wrt_T_W_S1.block<3, 3>(0, 0) = -q_W_S1.toRotationMatrix() *
          common::skew(p_S_U1);
      J_p_W_U1_wrt_T_W_S1.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
      J = sqrt_information_ * J_res_wrt_difference *
          J_p_W_U1_wrt_T_W_S1 * J_local;
    }

    if (jacobians[kIdxExtrinsics1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>
          J(jacobians[kIdxExtrinsics1]);
      Eigen::Matrix3d J_p_W_U1_wrt_p_S_U1 = q_W_S1.toRotationMatrix();
      J = sqrt_information_ * J_res_wrt_difference * J_p_W_U1_wrt_p_S_U1;
    }
  }

  return true;
}

} // namespace posegraph

} // namespace robopt
