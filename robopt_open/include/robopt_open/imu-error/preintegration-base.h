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
 * preintegration-base.h
 * @brief Header file to perform the imu pre-integration.
 * @author: Marco Karrer
 * Created on: Apr 24, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <robopt_open/common/common.h>
#include <robopt_open/common/typedefs.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace imu {

class PreintegrationBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreintegrationBase() = delete;

  /// \brief Construct a container for the imu-preintegration.
  /// @param init_acceleration Initial acceleration at which the int. starts.
  /// @param init_omega Initial body-rotation rate at which the int. start.
  /// @param linear_bias_a Linearized bias term for acceleration.
  /// @param linear_bias_g Linearized bias term for gyroscope.
  /// @param acc_nc Noise density of acceleremoter in continuos time.
  /// @param gyr_nc Noise density of gyroscope in continous time.
  /// @param acc_bc Noise density of cont. time random walk of accel. bias.
  /// @param gyr_bc Noise density of cont. time random walk of gyro bias.
  /// @param g_mag Magnitude of the gravity vector.
  PreintegrationBase(const Eigen::Vector3d& init_acceleration,
      const Eigen::Vector3d& init_omega,
      const Eigen::Vector3d& linear_bias_a,
      const Eigen::Vector3d& linear_bias_g,
      const double acc_nc, const double gyr_nc,
      const double acc_bc, const double gyr_bc,
      const double g_mag = 9.81);

  /// \brief Insert a new imu measurements in the preintegration.
  /// @param dt The time from the last measurement.
  /// @param acc_meas The linear acceleration measurement.
  /// @param gyr_meas The body-rotation rate measurement.
  void push_back(const double dt, const Eigen::Vector3d& acc_meas,
      const Eigen::Vector3d& gyr_meas);

  /// \brief Reintegrate the imu measurements with given bias.
  /// @param bias_a The acceleration bias.
  /// @param bias_g The gyroscope bias.
  void repropagate(const Eigen::Vector3d& bias_a,
                   const Eigen::Vector3d& bias_g);

  /// \brief Evaluate the residual of the preintegrated measurement.
  /// @param p_W_S1 The translation of the first pose.
  /// @param q_W_S1 The rotation of the first pose.
  /// @param v_S1 The velocity of the first frame.
  /// @param b_a1 The accelerometer bias of the first frame.
  /// @param b_g1 The gyroscope bias of the first frame.
  /// @param p_W_S2 The translation of the second pose.
  /// @param q_W_S2 The rotation of the second pose.
  /// @param v_S2 The velocity of the second frame.
  /// @param b_a2 The accelerometer bias of the second frame.
  /// @param b_g2 The gyroscope bias of the second frame.
  /// @return The 15x1 residual.
  Eigen::Matrix<double, 15, 1> evaluate(
      const Eigen::Vector3d& p_W_S1, const Eigen::Quaterniond& q_W_S1,
      const Eigen::Vector3d& v_S1, const Eigen::Vector3d& b_a1,
      const Eigen::Vector3d& b_g1,
      const Eigen::Vector3d& p_W_S2, const Eigen::Quaterniond& q_W_S2,
      const Eigen::Vector3d& v_S2, const Eigen::Vector3d& b_a2,
      const Eigen::Vector3d& b_g2);

  /// \brief Get the square root information matrix of the current state.
  /// @return The 15x15 square root information matrix.
  Eigen::Matrix<double, 15, 15> getSquareRootInformation();

  /// \brief Get the jacobian matrix of the current state.
  /// @return The 15x15 jacobian matrix.
  Eigen::Matrix<double, 15, 15> getJacobian() { return jacobian_;}

  /// \brief Get the linearized corrected transformation.
  /// @param v_S1 The velocity of the first frame.
  /// @param b_a1 The accelerometer bias of the first frame.
  /// @param b_g1 The gyroscope bias of the first frame.
  /// @param delta_p The corrected translation delta.
  /// @param delta_q The corrected rotation delta.
  /// @param delta_v The corrected velocity delta.
  void getDelta(const Eigen::Vector3d& p_W_S1,
      const Eigen::Quaterniond& q_W_S1, const Eigen::Vector3d& v_S1,
      const Eigen::Vector3d& b_a1, const Eigen::Vector3d& b_g1,
      Eigen::Vector3d* delta_p, Eigen::Quaterniond* delta_q,
      Eigen::Vector3d* delta_v);

  /// \brief Get the underlying stored deltas.
  /// @param delta_p The corrected translation delta.
  /// @param delta_q The corrected rotation delta.
  /// @param delta_v The corrected velocity delta.
  void getDelta(
      Eigen::Vector3d* delta_p, Eigen::Quaterniond* delta_q,
      Eigen::Vector3d* delta_v) const;

  /// \brief Get the accumulated time between the two frames.
  /// @return sum_dt The time difference.
  double getTimeSum() const { return sum_dt_; }

  /// \brief Get the gravity vector used for the preintegration.
  /// @return g The gravity vector.
  Eigen::Vector3d getGravity() const { return g_; }

  /// \brief Get the linearized biases.
  /// @param acc_bias The acceleration bias.
  /// @param gyr_bias The gyroscope bias.
  void getLinearizedBias(
      Eigen::Vector3d* acc_bias,
      Eigen::Vector3d* gyr_bias) const;

  /// \brief Get the number of buffered IMU measurements.
  /// @return The number of buffered IMU measurements.
  size_t getNumMeasurements() const { return acc_buf_.size(); }

  /// \brief Get a specific IMU measurement (by index).
  /// @param index The measurement index.
  /// @param acc The acceleration reading.
  /// @param gyr The gyroscope reading.
  /// @return Whether or not the measurement could be retrieved.
  bool getReadingsByIndex(
      const size_t index,
      Eigen::Vector3d* acc,
      Eigen::Vector3d* gyr) const;

  /// \brief Get a specific delta_t (by index).
  /// @param index The measurement index.
  /// @return The delta_t (-1.0 if out of bounds).
  double getTimeDiffByIndex(
      const size_t index) const;

protected:
  /// \brief Propagate the measurement
  /// @param dt The time from the last measurement.
  /// @param acc_meas The linear acceleration measurement.
  /// @param gyr_meas The body-rotation rate measurement.
  void propagate(const double dt, const Eigen::Vector3d& acc_meas,
      const Eigen::Vector3d& gyr_meas);

  /// \brief Perform IMU integration using midpoint integration.
  /// @param dt The time from the last measurement.
  /// @param result_delta_p The resulting new preint. delta in translation.
  /// @param result_delta_q The resulting new preint. delta in rotation.
  /// @param result_delta_v The resulting new preint. delta in velocity.
  /// @param update_jacobian Flag whether the jacobian should be updated.
  void midPointIntegration(double dt, Eigen::Vector3d* result_delta_p,
        Eigen::Quaterniond* result_delta_q, Eigen::Vector3d* result_delta_v,
        const bool update_jacobian = true);

  // Current state
  double dt_;
  Eigen::Vector3d acc_0_, gyr_0_;
  Eigen::Vector3d acc_1_, gyr_1_;

  // Initial measurements
  const Eigen::Vector3d init_acceleration_;
  const Eigen::Vector3d init_omega_;

  // Linearized biaas estimates
  Eigen::Vector3d linear_bias_a_;
  Eigen::Vector3d linear_bias_g_;

  // Covariance and jacobian matrices
  Eigen::Matrix<double, 15, 15> jacobian_, covariance_;
  Eigen::Matrix<double, 15, 15> step_jacobian_;
  Eigen::Matrix<double, 15, 18> step_V_;
  Eigen::Matrix<double, 18, 18> noise_c_;

  // The raw measurement buffers
  std::vector<double> dt_buf_;
  Vector3Vector acc_buf_;
  Vector3Vector gyr_buf_;

  // The preintegration measurments
  double sum_dt_;
  Eigen::Vector3d delta_p_;
  Eigen::Quaterniond delta_q_;
  Eigen::Vector3d delta_v_;

  // Sensor noise characteristics
  const double acc_nc_, acc_bc_;
  const double gyr_nc_, gyr_bc_;

  // Gravity information
  const Eigen::Vector3d g_;
  const double g_mag_;

public:
  FRIEND_TEST(PreintegrationBaseTerms, IntegrationJacobianState);
  FRIEND_TEST(PreintegrationBaseTerms, IntegrationJacobianNoise);
        
};

} // namespace imu

} // namespace robopt
