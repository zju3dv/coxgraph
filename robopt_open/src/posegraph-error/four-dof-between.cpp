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
 * four-dof-between.cpp
 * @brief Implementation file for relative pose constraints between two frames.
 * @author: Marco Karrer
 * Created on: Aug 18, 2018
 */

#include <posegraph-error/four-dof-between.h>

namespace robopt {

namespace posegraph {

bool FourDofBetweenError::Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const {
  // Coordinate frames:
  //  W = global
  //  S = IMU frame, expressed in W
  //  C = Camera frame, expressed in I
  //  1 = index of first reference
  //  2 = index of second reference

  // Unpack parameter blocks.
  Eigen::Map<const Eigen::Quaterniond> q_W_S1(parameters[kIdxPose1]);
  Eigen::Map<const Eigen::Vector3d> p_W_S1(parameters[kIdxPose1] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Quaterniond> q_W_S2(parameters[kIdxPose2]);
  Eigen::Map<const Eigen::Vector3d> p_W_S2(parameters[kIdxPose2] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Quaterniond> q_S_C1(parameters[kIdxExtrinsics1]);
  Eigen::Map<const Eigen::Vector3d> p_S_C1(parameters[kIdxExtrinsics1] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Quaterniond> q_S_C2(parameters[kIdxExtrinsics2]);
  Eigen::Map<const Eigen::Vector3d> p_S_C2(parameters[kIdxExtrinsics2] +
      defs::pose::kOrientationBlockSize);

  // Compute the residual
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>>
      residual(residuals);
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>
      error_term;

  if (error_term_type_ == defs::pose::PoseErrorType::kVisual) {
    // measurement is from C2 to C1
//    const Eigen::Quaterniond q_W_C1 = q_W_S1 * q_S_C1;
//    const double yaw_C1 = common::yaw::LogMap(q_W_C1);
//    Eigen::Vector3d p_W_C1 = q_W_S1 * p_S_C1 + p_W_S1;
//    Eigen::Quaterniond q_W_C2 = q_W_S2 * q_S_C2;
//    const double yaw_C2 = common::yaw::LogMap(q_W_C2);
//    Eigen::Vector3d p_W_C2 = q_W_S2 * p_S_C2 + p_W_S2;

//    residual[0] = common::yaw::normalizeYaw(yaw_C1 - yaw_C2 -
//        rotation_measurement_);
//    residual.tail<3>() = p_W_C1 - p_W_C2 - translation_measurement_;
//    error_term = residual;
//    residual = sqrt_information_ * residual;
    LOG(FATAL) << "Visual reference frame is not yet implemented";
    return false;
  } else if (error_term_type_ == defs::pose::PoseErrorType::kImu) {
    // measurement is from S2 to S1
    const double yaw_S1 = common::yaw::LogMap(q_W_S1);
    const double yaw_S2 = common::yaw::LogMap(q_W_S2);

    residual[0] = common::yaw::normalizeYaw(yaw_S1 - yaw_S2 -
        rotation_measurement_);
    residual.tail<3>() = p_W_S1 - p_W_S2 - translation_measurement_;
    error_term = residual;
    residual = sqrt_information_ * residual;
  } else {
     LOG(FATAL) << "Unknown pose error term type.";
  }

  // Compute the jacobians
  if (jacobians) {
    local_param::QuaternionLocalParameterization quat_local_parameterization;
    if (error_term_type_ == defs::pose::PoseErrorType::kVisual) {
      LOG(FATAL) << "Visual reference frame is not yet implemented";
//      const Eigen::Matrix3d R_W_S1 = q_W_S1.toRotationMatrix();
//      const Eigen::Matrix3d R_W_S2 = q_W_S2.toRotationMatrix();
//      const Eigen::Matrix3d R_S_C1 = q_S_C1.toRotationMatrix();
//      const Eigen::Matrix3d R_S_C2 = q_S_C2.toRotationMatrix();

//      const Eigen::Matrix3d R_W_C1 = R_W_S1 * R_S_C1;
//      const Eigen::Matrix3d R_W_C2 = R_W_S2 * R_S_C2;
//      const Eigen::Matrix3d R_C2_W = R_W_C2.transpose();
//      const Eigen::Vector3d p_W_C1 = R_W_S1 * p_S_C1 + p_W_S1;
//      const Eigen::Vector3d p_W_C2 = R_W_S2 * p_S_C2 + p_W_S2;
//      const Eigen::Matrix3d R_C2_C1 = R_C2_W * R_W_C1;
//      // const Eigen::Vector3d p_C2_C1 = R_C2_W * (p_W_C1 - p_W_C2);

//      // Compute some partial jacobians
//      /*PoseJacobianMin J_T_W_C1_wrt_T_W_S1;
//      J_T_W_C1_wrt_T_W_S1.setZero();
//      J_T_W_C1_wrt_T_W_S1.block<3,3>(0,0) = R_S_C1.transpose();
//      J_T_W_C1_wrt_T_W_S1.block<3,3>(3,0) = -R_W_S1 * common::skew(p_S_C1);
//      J_T_W_C1_wrt_T_W_S1.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
//      PoseJacobianMin J_T_W_C2_wrt_T_W_S2;
//      J_T_W_C2_wrt_T_W_S2.setZero();
//      J_T_W_C2_wrt_T_W_S2.block<3,3>(0,0) = R_S_C2.transpose();
//      J_T_W_C2_wrt_T_W_S2.block<3,3>(3,0) = -R_W_S2 * common::skew(p_S_C2);
//      J_T_W_C2_wrt_T_W_S2.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
//      PoseJacobianMin J_T_C2_C1_wrt_T_W_C1;
//      J_T_C2_C1_wrt_T_W_C1.setZero();
//      J_T_C2_C1_wrt_T_W_C1.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
//      J_T_C2_C1_wrt_T_W_C1.block<3,3>(3,3) = R_C2_W;
//      PoseJacobianMin J_T_C2_C1_wrt_T_W_C2;
//      J_T_C2_C1_wrt_T_W_C2.setZero();
//      J_T_C2_C1_wrt_T_W_C2.block<3,3>(0,0) = -R_C2_C1.transpose();
//      J_T_C2_C1_wrt_T_W_C2.block<3,3>(3,0) = common::skew(R_C2_W *
//          (p_W_C1 - p_W_C2));
//      J_T_C2_C1_wrt_T_W_C2.block<3,3>(3,3) = -R_C2_W;

//      Eigen::Matrix<double, defs::pose::kResidualSize, 6> J_res_wrt_T_C2_C1;
//      Eigen::Matrix3d gamma_T_C2_C1_inv =
//          common::quaternion::Gamma(error_term.head<3>()).inverse();
//      J_res_wrt_T_C2_C1.setZero();
//      J_res_wrt_T_C2_C1.block<3,3>(0,0) = gamma_T_C2_C1_inv *
//          rotation_measurement_.toRotationMatrix().transpose();
//      J_res_wrt_T_C2_C1.block<3,3>(3,0) = -R_C2_W * R_W_C1 *
//          common::skew(translation_measurement_);
//      J_res_wrt_T_C2_C1.block<3,3>(3,3) = -Eigen::Matrix3d::Identity();*/

//      if (jacobians[kIdxPose1]) {
//        Eigen::Map<PoseJacobian> J(jacobians[kIdxPose1]);
//        J.setZero();

//        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local;
//        quat_local_parameterization.ComputeJacobian(q_W_S1.coeffs().data(),
//              J_quat_local.data());
//        J.block<1,4>(0,0) = common::yaw::LogMapJacobian(
//              q_W_S1 * q_S_C1) * common::quaternion::OPlusMat(q_S_C1.coeffs());
//        J.block<3,4>(1,0) = - R_W_S1 * common::skew(p_S_C1) * 4.0 *
//            J_quat_local.transpose();
//        J.block<3,3>(1,4) = Eigen::Matrix3d::Identity();
//        J = sqrt_information_ * J;
//      }
//      if (jacobians[kIdxPose2]) {
//        Eigen::Map<PoseJacobian> J(jacobians[kIdxPose2]);
//        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
//        quat_parameterization.ComputeJacobian(
//            q_W_S2.coeffs().data(), J_quat_local_param.data());
//        Eigen::Matrix<double, defs::pose::kResidualSize,
//            defs::pose::kPoseBlockSize> J_local;
//        J_local.setZero();
//        J_local.block<3,4>(0,0) = J_quat_local_param.transpose() * 4.0;
//        J_local.block<3,3>(3,4) = Eigen::Matrix3d::Identity();
//        Eigen::Matrix<double, defs::pose::kResidualSize, 6> J_res_wrt_T_W_S2;
//        J_res_wrt_T_W_S2.setZero();
//        J_res_wrt_T_W_S2 = J_res_wrt_T_C2_C1 * J_T_C2_C1_wrt_T_W_C2 *
//            J_T_W_C2_wrt_T_W_S2;
//        J = sqrt_information_ * J_res_wrt_T_W_S2 * J_local;
//      }
//      if (jacobians[kIdxExtrinsics1]) {
//        Eigen::Map<PoseJacobian> J(jacobians[kIdxExtrinsics1]);
//        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
//        quat_parameterization.ComputeJacobian(
//            q_S_C1.coeffs().data(), J_quat_local_param.data());
//        PoseJacobianMin J_T_W_C1_wrt_T_S_C1;
//        J_T_W_C1_wrt_T_S_C1.setZero();
//        J_T_W_C1_wrt_T_S_C1.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
//        J_T_W_C1_wrt_T_S_C1.block<3,3>(3,3) = R_W_S1;
//        Eigen::Matrix<double, defs::pose::kResidualSize,
//            defs::pose::kPoseBlockSize> J_local;
//        J_local.setZero();
//        J_local.block<3,4>(0,0) = J_quat_local_param.transpose() * 4.0;
//        J_local.block<3,3>(3,4) = Eigen::Matrix3d::Identity();
//        Eigen::Matrix<double, defs::pose::kResidualSize, 6> J_res_wrt_T_S_C1;
//        J_res_wrt_T_S_C1.setZero();
//        J_res_wrt_T_S_C1 = J_res_wrt_T_C2_C1 * J_T_C2_C1_wrt_T_W_C1 *
//            J_T_W_C1_wrt_T_S_C1;
//        J = sqrt_information_ * J_res_wrt_T_S_C1 * J_local;
//      }
//      if (jacobians[kIdxExtrinsics2]) {
//        Eigen::Map<PoseJacobian> J(jacobians[kIdxExtrinsics2]);
//        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
//        quat_parameterization.ComputeJacobian(
//            q_S_C2.coeffs().data(), J_quat_local_param.data());
//        PoseJacobianMin J_T_W_C2_wrt_T_S_C2;
//        J_T_W_C2_wrt_T_S_C2.setZero();
//        J_T_W_C2_wrt_T_S_C2.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
//        J_T_W_C2_wrt_T_S_C2.block<3,3>(3,3) = R_W_S2;
//        Eigen::Matrix<double, defs::pose::kResidualSize,
//            defs::pose::kPoseBlockSize> J_local;
//        J_local.setZero();
//        J_local.block<3,4>(0,0) = J_quat_local_param.transpose() * 4.0;
//        J_local.block<3,3>(3,4) = Eigen::Matrix3d::Identity();
//        Eigen::Matrix<double, defs::pose::kResidualSize, 6> J_res_wrt_T_S_C2;
//        J_res_wrt_T_S_C2.setZero();
//        J_res_wrt_T_S_C2 = J_res_wrt_T_C2_C1 * J_T_C2_C1_wrt_T_W_C2 *
//            J_T_W_C2_wrt_T_S_C2;
//        J = sqrt_information_ * J_res_wrt_T_S_C2 * J_local;
//      }
    } else if (error_term_type_ == defs::pose::PoseErrorType::kImu) {
      if (jacobians[kIdxPose1]) {
        Eigen::Map<PoseJacobian> J(jacobians[kIdxPose1]);
        Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
            defs::pose::kPoseBlockSize> J_local;
        J_local.setZero();
        J_local.block<1,4>(0,0) = common::yaw::LogMapJacobian(q_W_S1);
        J_local.block<3,3>(1,4) = Eigen::Matrix3d::Identity();

        PoseYawJacobianMin J_res_wrt_T_W_S1;
        J_res_wrt_T_W_S1.setZero();
        J_res_wrt_T_W_S1(0,0) = 1.0;
        J_res_wrt_T_W_S1.block<3,3>(1,1) = Eigen::Matrix3d::Identity();
        J = sqrt_information_ * J_res_wrt_T_W_S1 * J_local;
      }
      if (jacobians[kIdxPose2]) {
        Eigen::Map<PoseJacobian> J(jacobians[kIdxPose2]);
        Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
            defs::pose::kPoseBlockSize> J_local;
        J_local.setZero();
        J_local.block<1,4>(0,0) = common::yaw::LogMapJacobian(q_W_S2);
        J_local.block<3,3>(1,4) = Eigen::Matrix3d::Identity();

        PoseYawJacobianMin J_res_wrt_T_W_S2;
        J_res_wrt_T_W_S2.setZero();
        J_res_wrt_T_W_S2(0,0) = -1.0;
        J_res_wrt_T_W_S2.block<3,3>(1,1) = -Eigen::Matrix3d::Identity();
        J = sqrt_information_ * J_res_wrt_T_W_S2 * J_local;
      }
      if (jacobians[kIdxExtrinsics1]) {
        Eigen::Map<PoseJacobian> J(jacobians[kIdxExtrinsics1]);
        J.setZero();
      }
      if (jacobians[kIdxExtrinsics2]) {
        Eigen::Map<PoseJacobian> J(jacobians[kIdxExtrinsics2]);
        J.setZero();
      }
    } /*else {
      LOG(FATAL) << "Unknown pose error term type.";
    }*/
  }

  return true;
}

} // namespace posegraph

} // namespace robopt
