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
 * preintegration-base.cpp
 * @brief Source file to perform the imu pre-integration.
 * @author: Marco Karrer
 * Created on: Apr 25, 2018
 */

#include <iostream>
#include <imu-error/preintegration-base.h>
#include <common/definitions.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace imu {

PreintegrationBase::PreintegrationBase(
    const Eigen::Vector3d &init_acceleration, const Eigen::Vector3d &init_omega,
    const Eigen::Vector3d &linear_bias_a, const Eigen::Vector3d &linear_bias_g,
    const double acc_nc, const double gyr_nc, const double acc_bc,
    const double gyr_bc, const double g_mag) :
      init_acceleration_(init_acceleration), init_omega_(init_omega),
      acc_0_(init_acceleration), gyr_0_(init_omega),
      linear_bias_a_(linear_bias_a), linear_bias_g_(linear_bias_g),
      acc_nc_(acc_nc), gyr_nc_(gyr_nc), acc_bc_(acc_bc), gyr_bc_(gyr_bc),
      g_mag_(g_mag), g_(Eigen::Vector3d(0.0, 0.0, g_mag)), sum_dt_(0.0) {
  // Initialize the matrices
  noise_c_ = Eigen::Matrix<double, 18, 18>::Zero();
  noise_c_.block<3,3>(0,0) = (acc_nc_ * acc_nc_) * Eigen::Matrix3d::Identity();
  noise_c_.block<3,3>(3,3) = (gyr_nc_ * gyr_nc_) * Eigen::Matrix3d::Identity();
  noise_c_.block<3,3>(6,6) = (acc_nc_ * acc_nc_) * Eigen::Matrix3d::Identity();
  noise_c_.block<3,3>(9,9) = (gyr_nc_ * gyr_nc_) * Eigen::Matrix3d::Identity();
  noise_c_.block<3,3>(12,12) = (acc_bc_ * acc_bc_) *
      Eigen::Matrix3d::Identity();
  noise_c_.block<3,3>(15,15) = (gyr_bc_ * gyr_bc_) *
      Eigen::Matrix3d::Identity();
  delta_p_.setZero();
  delta_q_.setIdentity();
  delta_v_.setZero();
  jacobian_.setIdentity();
  covariance_.setZero();
}

void PreintegrationBase::push_back(const double dt,
    const Eigen::Vector3d &acc_meas, const Eigen::Vector3d &gyr_meas) {
  // Store the measurement
  dt_buf_.push_back(dt);
  acc_buf_.push_back(acc_meas);
  gyr_buf_.push_back(gyr_meas);

  // Integrate the measurment
  propagate(dt, acc_meas, gyr_meas);
}

void PreintegrationBase::repropagate(const Eigen::Vector3d& bias_a,
    const Eigen::Vector3d& bias_g) {
  sum_dt_ = 0.0;
  acc_0_ = init_acceleration_;
  gyr_0_ = init_omega_;
  delta_p_.setZero();
  delta_q_.setIdentity();
  delta_v_.setZero();
  linear_bias_a_ = bias_a;
  linear_bias_g_ = bias_g;
  jacobian_.setIdentity();
  covariance_.setZero();
  for (size_t i = 0; i < dt_buf_.size(); ++i) {
    propagate(dt_buf_[i], acc_buf_[i], gyr_buf_[i]);
  }
}

Eigen::Matrix<double, 15, 1> PreintegrationBase::evaluate(
    const Eigen::Vector3d& p_W_S1, const Eigen::Quaterniond& q_W_S1,
    const Eigen::Vector3d& v_S1, const Eigen::Vector3d& b_a1,
    const Eigen::Vector3d& b_g1,
    const Eigen::Vector3d& p_W_S2, const Eigen::Quaterniond& q_W_S2,
    const Eigen::Vector3d& v_S2, const Eigen::Vector3d& b_a2,
    const Eigen::Vector3d& b_g2) {
  Eigen::Matrix<double, 15, 1> residuals;

  // Extract the partial jacobians needed for the evaluation
  Eigen::Matrix3d dp_dba = jacobian_.block<3,3>(
        defs::pose::StateOrder::kPosition, defs::pose::StateOrder::kBiasA);
  Eigen::Matrix3d dp_dbg = jacobian_.block<3,3>(
        defs::pose::StateOrder::kPosition, defs::pose::StateOrder::kBiasG);
  Eigen::Matrix3d dq_dbg = jacobian_.block<3,3>(
        defs::pose::StateOrder::kRotation, defs::pose::StateOrder::kBiasG);
  Eigen::Matrix3d dv_dba = jacobian_.block<3,3>(
        defs::pose::StateOrder::kVelocity, defs::pose::StateOrder::kBiasA);
  Eigen::Matrix3d dv_dbg = jacobian_.block<3,3>(
        defs::pose::StateOrder::kVelocity, defs::pose::StateOrder::kBiasG);

  Eigen::Vector3d dba = b_a1 - linear_bias_a_;
  Eigen::Vector3d dbg = b_g1 - linear_bias_g_;

  Eigen::Quaterniond exact_sum;
  common::quaternion::Plus(delta_q_, dq_dbg * dbg, &exact_sum);
  Eigen::Quaterniond corrected_delta_q = exact_sum;
      //delta_q_ * common::quaternion::DeltaQ(dq_dbg * dbg);
  Eigen::Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
  Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

  residuals.block<3,1>(defs::pose::StateOrder::kPosition, 0) =
      q_W_S1.inverse() *(0.5 * g_ * sum_dt_ * sum_dt_ + p_W_S2 - p_W_S1 -
      v_S1 * sum_dt_) - corrected_delta_p;
  Eigen::Vector3d rot_diff;
  Eigen::Quaterniond q_S1_S2 = q_W_S1.inverse()*q_W_S2;
  if (q_S1_S2.w() < 0) {
    q_S1_S2.w() = -q_S1_S2.w();
    q_S1_S2.x() = -q_S1_S2.x();
    q_S1_S2.y() = -q_S1_S2.y();
    q_S1_S2.z() = -q_S1_S2.z();
  }

  common::quaternion::Minus(q_S1_S2, corrected_delta_q,
      &rot_diff);
  residuals.block<3,1>(defs::pose::StateOrder::kRotation, 0) = rot_diff;
  residuals.block<3,1>(defs::pose::StateOrder::kVelocity, 0) =
      q_W_S1.inverse() * (g_ * sum_dt_ + v_S2 - v_S1) - corrected_delta_v;
  residuals.block<3,1>(defs::pose::StateOrder::kBiasA, 0) = b_a2 - b_a1;
  residuals.block<3,1>(defs::pose::StateOrder::kBiasG, 0) = b_g2 - b_g1;
  return residuals;
}

Eigen::Matrix<double, 15, 15> PreintegrationBase::getSquareRootInformation() {
  Eigen::Matrix<double, 15, 15> information = covariance_.inverse();
  return Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
            information).matrixL().transpose();
}

void PreintegrationBase::getDelta(const Eigen::Vector3d& p_W_S1,
    const Eigen::Quaterniond& q_W_S1, const Eigen::Vector3d& v_S1,
    const Eigen::Vector3d& b_a1, const Eigen::Vector3d& b_g1,
    Eigen::Vector3d* delta_p, Eigen::Quaterniond* delta_q,
    Eigen::Vector3d* delta_v) {
  // Extract the partial jacobians needed for the evaluation
  Eigen::Matrix3d dp_dba = jacobian_.block<3,3>(
        defs::pose::StateOrder::kPosition, defs::pose::StateOrder::kBiasA);
  Eigen::Matrix3d dp_dbg = jacobian_.block<3,3>(
        defs::pose::StateOrder::kPosition, defs::pose::StateOrder::kBiasG);
  Eigen::Matrix3d dq_dbg = jacobian_.block<3,3>(
        defs::pose::StateOrder::kRotation, defs::pose::StateOrder::kBiasG);
  Eigen::Matrix3d dv_dba = jacobian_.block<3,3>(
        defs::pose::StateOrder::kVelocity, defs::pose::StateOrder::kBiasA);
  Eigen::Matrix3d dv_dbg = jacobian_.block<3,3>(
        defs::pose::StateOrder::kVelocity, defs::pose::StateOrder::kBiasG);

  Eigen::Vector3d dba = b_a1 - linear_bias_a_;
  Eigen::Vector3d dbg = b_g1 - linear_bias_g_;

  if (delta_p) {
    *delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg -
        q_W_S1.inverse() * (0.5 * g_ * sum_dt_ * sum_dt_ - v_S1 * sum_dt_);
  }
  if (delta_q) {
    *delta_q = delta_q_ * common::quaternion::DeltaQ(dq_dbg * dbg);
  }
  if (delta_v) {
    *delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg -
        q_W_S1.inverse() * (g_ * sum_dt_);
  }
}

void PreintegrationBase::getDelta(
    Eigen::Vector3d* delta_p, Eigen::Quaterniond* delta_q,
    Eigen::Vector3d* delta_v) const {
  CHECK_NOTNULL(delta_p);
  CHECK_NOTNULL(delta_q);
  CHECK_NOTNULL(delta_v);
  *delta_p = delta_p_;
  *delta_q = delta_q_;
  *delta_v = delta_v_;
}

void PreintegrationBase::getLinearizedBias(
    Eigen::Vector3d* acc_bias,
    Eigen::Vector3d* gyr_bias) const {
  CHECK_NOTNULL(acc_bias);
  CHECK_NOTNULL(gyr_bias);

  *acc_bias = linear_bias_a_;
  *gyr_bias = linear_bias_g_;
}

bool PreintegrationBase::getReadingsByIndex(
    const size_t index,
    Eigen::Vector3d* acc,
    Eigen::Vector3d* gyr) const {
  CHECK_NOTNULL(acc);
  CHECK_NOTNULL(gyr);
  if (index >= acc_buf_.size()) {
    return false;
  }

  *acc = acc_buf_[index];
  *gyr = gyr_buf_[index];
  return true;
}

double PreintegrationBase::getTimeDiffByIndex(const size_t index) const {
  if (index >= acc_buf_.size()) {
    return false;
  }

  return dt_buf_[index];
}

void PreintegrationBase::propagate(const double dt,
    const Eigen::Vector3d &acc_meas, const Eigen::Vector3d &gyr_meas) {
  dt_ = dt;
  acc_1_ = acc_meas;
  gyr_1_ = gyr_meas;
  Eigen::Vector3d result_delta_p;
  Eigen::Quaterniond result_delta_q;
  Eigen::Vector3d result_delta_v;
  midPointIntegration(dt, &result_delta_p, &result_delta_q,
      &result_delta_v, true);
  delta_p_ = result_delta_p;
  delta_q_ = result_delta_q;
  delta_v_ = result_delta_v;
  delta_q_.normalize();
  sum_dt_ += dt;
  acc_0_ = acc_1_;
  gyr_0_ = gyr_1_;
}

void PreintegrationBase::midPointIntegration(double dt,
      Eigen::Vector3d* result_delta_p, Eigen::Quaterniond* result_delta_q,
      Eigen::Vector3d* result_delta_v, const bool update_jacobian) {
  // Compute the bias corrected accelerations & velocities
  Eigen::Vector3d un_acc_0 = delta_q_ * (acc_0_ - linear_bias_a_);
  Eigen::Vector3d un_gyr = 0.5 * (gyr_0_ + gyr_1_) - linear_bias_g_;
  Eigen::Quaterniond un_gyr_quat = Eigen::Quaterniond(
        1.0, un_gyr(0) * dt/2.0, un_gyr(1) * dt/2.0, un_gyr(2) * dt/2.0);
  un_gyr_quat.normalize();
  *result_delta_q = delta_q_ * un_gyr_quat;
  Eigen::Vector3d un_acc_1 = (*result_delta_q) * (acc_1_ - linear_bias_a_);
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  *result_delta_p = delta_p_ + delta_v_ * dt + 0.5 * un_acc * dt * dt;
  *result_delta_v = delta_v_ + un_acc * dt;

  if (update_jacobian) {
    const Eigen::Vector3d a_0_x = acc_0_ - linear_bias_a_;
    const Eigen::Vector3d a_1_x = acc_1_ - linear_bias_a_;
    const Eigen::Matrix3d R_a_0_x = common::skew(a_0_x);
    const Eigen::Matrix3d R_a_1_x = common::skew(a_1_x);
    const Eigen::Matrix3d R_result_delta_q = result_delta_q->toRotationMatrix();
    const Eigen::Matrix3d R_delta_q = delta_q_.toRotationMatrix();
    const Eigen::Matrix3d gamma_un_gyr = common::quaternion::Gamma(un_gyr * dt);
    const Eigen::Matrix3d R_un_gyr = un_gyr_quat.toRotationMatrix();

    // Jacobian w.r.t. state
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
    F.block<3,3>(defs::pose::StateOrder::kPosition,
        defs::pose::StateOrder::kPosition) = Eigen::Matrix3d::Identity();
    F.block<3,3>(defs::pose::StateOrder::kPosition,
        defs::pose::StateOrder::kRotation) = -0.25 *
        delta_q_.toRotationMatrix() * R_a_0_x * dt * dt - 0.25 *
        R_result_delta_q * R_a_1_x * R_un_gyr.transpose() * dt * dt;
    F.block<3,3>(defs::pose::StateOrder::kPosition,
        defs::pose::StateOrder::kVelocity) = Eigen::Matrix3d::Identity() * dt;
    F.block<3,3>(defs::pose::StateOrder::kPosition,
        defs::pose::StateOrder::kBiasA) =
        -0.25 * (R_delta_q + R_result_delta_q) * dt * dt;
    F.block<3,3>(defs::pose::StateOrder::kPosition,
        defs::pose::StateOrder::kBiasG) =
        0.25 * R_result_delta_q * R_a_1_x * dt * dt * dt;
    F.block<3,3>(defs::pose::StateOrder::kRotation,
        defs::pose::StateOrder::kRotation) =
        R_un_gyr.transpose();
    F.block<3,3>(defs::pose::StateOrder::kRotation,
        defs::pose::StateOrder::kBiasG) =
        -1.0 * gamma_un_gyr * dt;
    F.block<3,3>(defs::pose::StateOrder::kVelocity,
        defs::pose::StateOrder::kRotation) = -0.5 * R_delta_q * R_a_0_x * dt -
        0.5 * R_result_delta_q * R_a_1_x * R_un_gyr.transpose() * dt;
    F.block<3,3>(defs::pose::StateOrder::kVelocity,
        defs::pose::StateOrder::kVelocity) = Eigen::Matrix3d::Identity();
    F.block<3,3>(defs::pose::StateOrder::kVelocity,
        defs::pose::StateOrder::kBiasA) =
        -0.5 * (R_delta_q + R_result_delta_q) * dt;
    F.block<3,3>(defs::pose::StateOrder::kVelocity,
        defs::pose::StateOrder::kBiasG) =
        0.5 * R_result_delta_q * R_a_1_x * dt * dt;
    F.block<3,3>(defs::pose::StateOrder::kBiasA,
        defs::pose::StateOrder::kBiasA) = Eigen::Matrix3d::Identity();
    F.block<3,3>(defs::pose::StateOrder::kBiasG,
        defs::pose::StateOrder::kBiasG) = Eigen::Matrix3d::Identity();

    // Jacobian w.r.t. noise
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
    V.block<3,3>(defs::pose::StateOrder::kPosition,0) =
        0.25 * R_delta_q * dt * dt;
    V.block<3,3>(defs::pose::StateOrder::kPosition,3) =
        -0.25 * R_result_delta_q * R_a_1_x  * dt * dt * 0.5 * dt;
    V.block<3,3>(defs::pose::StateOrder::kPosition,6) =
        0.25 * R_result_delta_q * dt * dt;
    V.block<3,3>(defs::pose::StateOrder::kPosition,9) = V.block<3, 3>(0, 3);
    V.block<3,3>(defs::pose::StateOrder::kRotation,3) =
        0.5 * Eigen::Matrix3d::Identity() * dt;
    V.block<3,3>(defs::pose::StateOrder::kRotation,9) =
        0.5 * Eigen::Matrix3d::Identity() * dt;
    V.block<3,3>(defs::pose::StateOrder::kVelocity,0) = 0.5 * R_delta_q * dt;
    V.block<3,3>(defs::pose::StateOrder::kVelocity,3) =
        -0.5 * R_result_delta_q * R_a_1_x  * dt * 0.5 * dt;
    V.block<3,3>(defs::pose::StateOrder::kVelocity,6) =
        0.5 * R_result_delta_q * dt;
    V.block<3,3>(defs::pose::StateOrder::kVelocity,9) = V.block<3,3>(6,3);
    V.block<3,3>(defs::pose::StateOrder::kBiasA,12) =
        Eigen::Matrix3d::Identity() * dt;
    V.block<3,3>(defs::pose::StateOrder::kBiasG,15) =
        Eigen::Matrix3d::Identity() * dt;

    step_jacobian_ = F;
    jacobian_ = F * jacobian_;
    covariance_ = F * covariance_ * F.transpose() +
        V * noise_c_ * V.transpose();
    step_V_ = V;
  }
}


} // namespace imu

} // namespace robopt
