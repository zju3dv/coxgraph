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
 * preintegration-factor.cpp
 * @brief Source file for the imu-preintegration factor.
 * @author: Marco Karrer
 * Created on: Jun 7, 2018
 */

#include <imu-error/preintegration-factor.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace imu {

PreintegrationFactor::PreintegrationFactor(PreintegrationBase* preint) :
    pre_integration_(preint)
{

}

bool PreintegrationFactor::Evaluate(const double * const *parameters,
    double *residuals, double **jacobians) const {
  // Extract the parameters
  Eigen::Map<const Eigen::Quaterniond> q_W_S1(parameters[kIdxPose1]);
  Eigen::Map<const Eigen::Vector3d> p_W_S1(parameters[kIdxPose1] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Vector3d> v_S1(parameters[kIdxSpeedBias1]);
  Eigen::Map<const Eigen::Vector3d> b_a1(parameters[kIdxSpeedBias1] + 3);
  Eigen::Map<const Eigen::Vector3d> b_g1(parameters[kIdxSpeedBias1] + 6);
  Eigen::Map<const Eigen::Quaterniond> q_W_S2(parameters[kIdxPose2]);
  Eigen::Map<const Eigen::Vector3d> p_W_S2(parameters[kIdxPose2] +
      defs::pose::kOrientationBlockSize);
  Eigen::Map<const Eigen::Vector3d> v_S2(parameters[kIdxSpeedBias2]);
  Eigen::Map<const Eigen::Vector3d> b_a2(parameters[kIdxSpeedBias2] + 3);
  Eigen::Map<const Eigen::Vector3d> b_g2(parameters[kIdxSpeedBias2] + 6);

  // Compute the residuals
  Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
  Eigen::Matrix<double, 15, 1> error_term = pre_integration_->evaluate(
        p_W_S1, q_W_S1, v_S1, b_a1, b_g1,
      p_W_S2, q_W_S2, v_S2, b_a2, b_g2);
  Eigen::Matrix<double, 15, 15> sqrt_info =
      pre_integration_->getSquareRootInformation();
  residual = sqrt_info * error_term;

  if (jacobians) {
    // Extract some useful blocks
    const double sum_dt = pre_integration_->getTimeSum();
    const Eigen::Matrix<double, 15, 15> J_state =
        pre_integration_->getJacobian();

    const Eigen::Matrix3d dp_dba = J_state.block<3,3>(
          defs::pose::StateOrder::kPosition, defs::pose::StateOrder::kBiasA);
    const Eigen::Matrix3d dp_dbg = J_state.block<3,3>(
          defs::pose::StateOrder::kPosition, defs::pose::StateOrder::kBiasG);

    const Eigen::Matrix3d dq_dbg = J_state.block<3,3>(
          defs::pose::StateOrder::kRotation, defs::pose::StateOrder::kBiasG);

    const Eigen::Matrix3d dv_dba = J_state.block<3,3>(
          defs::pose::StateOrder::kVelocity, defs::pose::StateOrder::kBiasA);
    const Eigen::Matrix3d dv_dbg = J_state.block<3,3>(
          defs::pose::StateOrder::kVelocity, defs::pose::StateOrder::kBiasG);
    const Eigen::Matrix3d R_W_S1 = q_W_S1.toRotationMatrix();
    const Eigen::Matrix3d R_W_S2 = q_W_S2.toRotationMatrix();
    const Eigen::Vector3d g = pre_integration_->getGravity();
    const Eigen::Matrix3d gamma_rot_err_inv =
        common::quaternion::Gamma(error_term.segment<3>(
        defs::pose::StateOrder::kRotation)).inverse();

    if (J_state.maxCoeff() > 1e8 || J_state.minCoeff() < -1e8) {
      std::cout << "Numerical unstable preintegration factor" << std::endl;
    }

    if (jacobians[kIdxPose1]) {
      local_param::QuaternionLocalParameterization quat_parameterization;
      Eigen::Map<PoseJacobian> J_res_wrt_T_W_S1(jacobians[kIdxPose1]);
      J_res_wrt_T_W_S1.setZero();
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parameterization.ComputeJacobian(
          q_W_S1.coeffs().data(), J_quat_local_param.data());
      Eigen::Matrix<double, 3, 4> J_local =
          J_quat_local_param.transpose() * 4.0;

      // Derivative w.r.t. q_W_S1
      J_res_wrt_T_W_S1.block<3, defs::pose::kOrientationBlockSize>(
            defs::pose::StateOrder::kRotation, 0) =
          -gamma_rot_err_inv * R_W_S2.transpose() * R_W_S1 * J_local;
      J_res_wrt_T_W_S1.block<3, defs::pose::kOrientationBlockSize>(
            defs::pose::StateOrder::kPosition, 0) =
          common::skew(R_W_S1.transpose() * (0.5 * g * sum_dt * sum_dt +
          p_W_S2 - p_W_S1 - v_S1 * sum_dt)) * J_local;
      J_res_wrt_T_W_S1.block<3, defs::pose::kOrientationBlockSize>(
            defs::pose::StateOrder::kVelocity, 0) =
          common::skew(R_W_S1.transpose() * (g * sum_dt + v_S2 - v_S1)) *
          J_local;

      // Derivative w.r.t. p_W_S1
      J_res_wrt_T_W_S1.block<3, defs::pose::kPositionBlockSize>(
            defs::pose::StateOrder::kPosition,
            defs::pose::kOrientationBlockSize) =
          -R_W_S1.transpose();

      J_res_wrt_T_W_S1 = sqrt_info * J_res_wrt_T_W_S1;
    }
    if (jacobians[kIdxSpeedBias1]) {
      Eigen::Map<SpeedBiasJacobian>
          J_res_wrt_vb1(jacobians[kIdxSpeedBias1]);
      J_res_wrt_vb1.setZero();

      // Derivative w.r.t. v_S1
      J_res_wrt_vb1.block<3, 3>(defs::pose::StateOrder::kPosition,
              defs::pose::StateOrder::kVelocity -
              defs::pose::StateOrder::kVelocity) = -R_W_S1.transpose() * sum_dt;
      J_res_wrt_vb1.block<3, 3>(defs::pose::StateOrder::kVelocity,
              defs::pose::StateOrder::kVelocity -
              defs::pose::StateOrder::kVelocity) = -R_W_S1.transpose();

      // Derivatives w.r.t. b_A1
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kPosition,
              defs::pose::StateOrder::kBiasA -
              defs::pose::StateOrder::kVelocity) = -dp_dba;
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kVelocity,
              defs::pose::StateOrder::kBiasA -
              defs::pose::StateOrder::kVelocity) = -dv_dba;
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kBiasA,
              defs::pose::StateOrder::kBiasA -
              defs::pose::StateOrder::kVelocity) = -Eigen::Matrix3d::Identity();

      // Derivatives w.r.t. b_G1
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kRotation,
              defs::pose::StateOrder::kBiasG -
              defs::pose::StateOrder::kVelocity) =
          -gamma_rot_err_inv * (R_W_S2.transpose() * R_W_S1) * dq_dbg.transpose();
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kPosition,
              defs::pose::StateOrder::kBiasG -
              defs::pose::StateOrder::kVelocity) = -dp_dbg;
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kVelocity,
              defs::pose::StateOrder::kBiasG -
              defs::pose::StateOrder::kVelocity) = -dv_dbg;
      J_res_wrt_vb1.block<3,3>(defs::pose::StateOrder::kBiasG,
              defs::pose::StateOrder::kBiasG -
              defs::pose::StateOrder::kVelocity) = -Eigen::Matrix3d::Identity();

      J_res_wrt_vb1 = sqrt_info * J_res_wrt_vb1;
    }
    if (jacobians[kIdxPose2]) {
      local_param::QuaternionLocalParameterization quat_parameterization;
      Eigen::Map<PoseJacobian>
          J_res_wrt_T_W_S2(jacobians[kIdxPose2]);
      J_res_wrt_T_W_S2.setZero();
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
      quat_parameterization.ComputeJacobian(
          q_W_S2.coeffs().data(), J_quat_local_param.data());
      Eigen::Matrix<double, 3, 4> J_local =
          J_quat_local_param.transpose() * 4.0;

      // Derivatives w.r.t. q_W_S2
      J_res_wrt_T_W_S2.block<3, defs::pose::kOrientationBlockSize>(
            defs::pose::StateOrder::kRotation, 0) =
          gamma_rot_err_inv * J_local;

      // Derivatives w.r.t. p_W_S2
      J_res_wrt_T_W_S2.block<3,defs::pose::kPositionBlockSize>(
            defs::pose::StateOrder::kPosition,
            defs::pose::kOrientationBlockSize) =
          R_W_S1.transpose();

      J_res_wrt_T_W_S2 = sqrt_info * J_res_wrt_T_W_S2;
    }
    if (jacobians[kIdxSpeedBias2]) {
      Eigen::Map<SpeedBiasJacobian> J_res_wrt_vb2(jacobians[kIdxSpeedBias2]);
      J_res_wrt_vb2.setZero();

      // Derivative w.r.t. v_S2
      J_res_wrt_vb2.block<3, 3>(defs::pose::StateOrder::kVelocity,
              defs::pose::StateOrder::kVelocity -
              defs::pose::StateOrder::kVelocity) = R_W_S1.transpose();

      // Derivatives w.r.t. b_A2
      J_res_wrt_vb2.block<3,3>(defs::pose::StateOrder::kBiasA,
              defs::pose::StateOrder::kBiasA -
              defs::pose::StateOrder::kVelocity) = Eigen::Matrix3d::Identity();

      // Derivatives w.r.t. b_G2
      J_res_wrt_vb2.block<3,3>(defs::pose::StateOrder::kBiasG,
              defs::pose::StateOrder::kBiasG -
              defs::pose::StateOrder::kVelocity) = Eigen::Matrix3d::Identity();

      J_res_wrt_vb2 = sqrt_info * J_res_wrt_vb2;
    }
  }
  
  return true;
}

} // namespace imu

} // namespace robopt
