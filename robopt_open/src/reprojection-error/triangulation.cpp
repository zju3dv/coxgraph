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
 * triangulation.cpp
 * @brief Implementation file for triangulation functionality.
 * @author: Marco Karrer
 * Created on: Nov 15, 2018
 */

#include <reprojection-error/triangulation.h>
#include <common/common.h>
#include <iostream>

namespace robopt {

namespace reprojection {

TriangulationStatus svdTriangulation(
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
      T_W_Si,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
      T_S_Ci,
  const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
      meas_normalized,
  Eigen::Vector3d* l_W) {
  if (meas_normalized.size() < 2) {
    return TriangulationStatus::kTooFewMeasurments;
  }

  // Initialize the data for the SVD
  const size_t rows = 3 * meas_normalized.size();
  const size_t cols = 3 + meas_normalized.size();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

  for (size_t i = 0; i < meas_normalized.size(); ++i) {
    Eigen::Vector3d v(meas_normalized[i](0), meas_normalized[i](1), 1.0);
    A.block<3,3>(3 * i, 0) = Eigen::Matrix3d::Identity();
    A.block<3,1>(3 * i, 3 + i) = -T_W_Si[i].block<3,3>(0, 0) *
        T_S_Ci[i].block<3,3>(0, 0) * v;
    b.segment<3>(3 * i) = T_W_Si[i].block<3,1>(0, 3) +
        T_W_Si[i].block<3,3>(0, 0) * T_S_Ci[i].block<3,1>(0, 3);
  }
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr = A.colPivHouseholderQr();

  // Check whether or not the point is well defined
  qr.setThreshold(0.001);
  const size_t rank = qr.rank();

  if ((rank - meas_normalized.size()) < 3) {
    return TriangulationStatus::kUnobservable;
  }

  *l_W = qr.solve(b).head<3>();

  return TriangulationStatus::kSuccessful;
}


TriangulationStatus nonlinearTriangulation(
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
    T_W_Si,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
    T_S_Ci,
  const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
    meas_normalized,
  const std::vector<double>& sqrt_info,
  Eigen::Vector3d* l_W,
  std::vector<bool>& outliers,
  const double c) {

  // Store the number of measurements
  const size_t num_meas = meas_normalized.size();

  if (num_meas < 2) {
    return TriangulationStatus::kTooFewMeasurments;
  }

  // Set parameters to determine convergence
  const double kPrecision = 1.0e-6;
  const size_t kIterMax = 10;

  // Initialize variables that are constant throughout the iterations
  const Eigen::Matrix4d T_C_W_ref = T_S_Ci[0].inverse() * T_W_Si[0].inverse();
  const Eigen::Matrix4d T_W_C_ref = T_C_W_ref.inverse();
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      R_Ci_Cref;
  R_Ci_Cref.reserve(num_meas);
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      p_Ci_Cref;
  p_Ci_Cref.reserve(num_meas);
  for (size_t i = 0; i < num_meas; ++i) {
    // Get the current transformation world to camera
    const Eigen::Matrix4d T_Ci_W = T_S_Ci[i].inverse() * T_W_Si[i].inverse();

    // Compute the Transformation from reference camera to current camera
    const Eigen::Matrix4d T_Ci_Cref = T_Ci_W * T_W_C_ref;


    // Rotation from first camera to current camera.
    R_Ci_Cref.emplace_back(T_Ci_Cref.block<3,3>(0, 0));

    // Translation from first camera to current camera.
    p_Ci_Cref.emplace_back(T_Ci_Cref.block<3,1>(0, 3));
  }

  // Transform the landmark into the reference camera frame
  Eigen::Vector3d l_Cref = T_C_W_ref.block<3,3>(0, 0) * (*l_W) +
      T_C_W_ref.block<3,1>(0, 3);

  // Initialize minimization variables.
  double alpha = l_Cref(0)/l_Cref(2); //0.0;
  double beta = l_Cref(1)/l_Cref(2); //0.0;
  double rho = 1.0/l_Cref(2); //0.0;
  double residual_norm_last = 1000.0;
  double residual_norm = 100.0;


  // Loop over iterations
  size_t iter = 0;
  while (residual_norm_last - residual_norm > kPrecision && iter < kIterMax) {
    Eigen::VectorXd residuals(num_meas * 2);
    Eigen::MatrixXd jacobian(num_meas * 2, 3);
    residuals.setZero();
    jacobian.setZero();

    // Loop over the measurements (camera frames)
    for (size_t i = 0; i < num_meas; ++i) {
      const Eigen::Vector2d h_meas = meas_normalized[i];

      // Predict measurements
      const Eigen::Vector3d h_i = R_Ci_Cref[i] *
          (Eigen::Matrix<double, 3, 1>() << alpha, beta, 1.0).finished() +
          rho * p_Ci_Cref[i];
      const Eigen::Vector2d h = h_i.head<2>()/h_i(2);

      // Calculate the residual
      double resid_norm = (h_meas - h).norm() * sqrt_info[i];

      double weight = std::sqrt(common::weighting::cauchyWeight(resid_norm, c));
      residuals.segment<2>(i * 2) = (h_meas - h) * weight * sqrt_info[i];

      if (weight > 0.8 * c) {
        outliers[i] = false;
      } else {
        outliers[i] = true;
      }

      // Calculate the jacobian
      Eigen::Matrix<double, 2, 3> jacobian_perspective;
      jacobian_perspective << -1.0 / h_i(2), 0.0, h_i(0) / (h_i(2) * h_i(2)),
      0.0, -1.0 / h_i(2), h_i(1) / (h_i(2) * h_i(2));

      const Eigen::Matrix<double, 3, 1> jacobian_alpha = R_Ci_Cref[i] *
          (Eigen::Matrix<double, 3, 1>() << 1.0, 0.0, 0.0).finished();
      const Eigen::Matrix<double, 3, 1> jacobian_beta = R_Ci_Cref[i] *
          (Eigen::Matrix<double, 3, 1>() << 0.0, 1.0, 0.0).finished();
      const Eigen::Matrix<double, 3, 1> jacobian_rho = p_Ci_Cref[i];

      const Eigen::Matrix<double, 2, 1> jacobian_A =
          jacobian_perspective * jacobian_alpha;
      const Eigen::Matrix<double, 2, 1> jacobian_B =
          jacobian_perspective * jacobian_beta;
      const Eigen::Matrix<double, 2, 1> jacobian_C =
          jacobian_perspective * jacobian_rho;

      jacobian.block<1,3>(i * 2, 0) = (Eigen::Matrix<double, 1, 3>() <<
           jacobian_A(0), jacobian_B(0), jacobian_C(0)).finished() * weight *
            sqrt_info[i];
      jacobian.block<1,3>(i * 2 + 1, 0) = (Eigen::Matrix<double, 1, 3>() <<
           jacobian_A(1), jacobian_B(1), jacobian_C(1)).finished() * weight *
            sqrt_info[i];
    }

    // Calculate update using LDLT decomposition.
    Eigen::Vector3d delta = (jacobian.transpose() * jacobian +
                             0.0001 * Eigen::Matrix3d::Identity())
        .ldlt().solve(jacobian.transpose() * residuals);

    alpha = alpha - delta(0);
    beta = beta - delta(1);
    rho = rho - delta(2);


    residual_norm_last = residual_norm;
    residual_norm = residuals.squaredNorm();
    ++iter;
  }

  if (rho < 0.0) {
    return TriangulationStatus::kUnobservable;
  }

  // Coordinate of feature in global frame.
  *l_W = 1.0/rho * T_W_C_ref.block<3,3>(0, 0) *
      (Eigen::Matrix<double, 3, 1>() << alpha, beta, 1.0).finished() +
      T_W_C_ref.block<3,1>(0, 3);

  return TriangulationStatus::kSuccessful;
}

} // namespace reprojection

} // namespace robopt
