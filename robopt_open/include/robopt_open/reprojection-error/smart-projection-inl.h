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
 * smart-projection-inl.h
 * @brief Implementation file for smart-reprojection factor.
 * @author: Marco Karrer
 * Created on: Nov 6, 2018
 */

/// \brief robopt Main namespace of this package
namespace robopt {

namespace reprojection {

template<typename CameraType, typename DistortionType>
bool SmartProjectionError<CameraType, DistortionType>::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  // Unpack the parameter blocks
  VectorConstMapQuaternion q_W_Si;
  q_W_Si.reserve(num_cams_);
  VectorConstMapVector3 p_W_Si;
  p_W_Si.reserve(num_cams_);
  VectorConstMapQuaternion q_S_Ci;
  q_S_Ci.reserve(num_cams_);
  VectorConstMapVector3 p_S_Ci;
  p_S_Ci.reserve(num_cams_);
  VectorMatrix4 T_W_Si;
  T_W_Si.reserve(num_cams_);
  VectorMatrix4 T_S_Ci;
  T_S_Ci.reserve(num_cams_);
  const size_t parameter_count = 2;
  std::vector<double> sqrt_infos(num_cams_, 0.0);
  for (size_t i = 0; i < num_cams_; ++i) {
    const size_t parameter_inc = i * parameter_count;
    Eigen::Map<const Eigen::Quaterniond> q_W_S(parameters[kIdxImuPose +
        parameter_inc]);
    q_W_Si.push_back(q_W_S);
    Eigen::Map<const Eigen::Vector3d> p_W_S(parameters[kIdxImuPose +
        parameter_inc] + defs::visual::kOrientationBlockSize);
    p_W_Si.push_back(p_W_S);
    Eigen::Map<const Eigen::Quaterniond> q_S_C(parameters[kIdxCameraToImu +
        parameter_inc]);
    q_S_Ci.push_back(q_S_C);
    Eigen::Map<const Eigen::Vector3d> p_S_C(parameters[kIdxCameraToImu +
        parameter_inc] + defs::visual::kOrientationBlockSize);
    p_S_Ci.push_back(p_S_C);

    // Also store the matrices
    Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
    T_W_S.block<3,3>(0, 0) = q_W_S.toRotationMatrix();
    T_W_S.block<3,1>(0, 3) = p_W_S;
    T_W_Si.push_back(T_W_S);
    Eigen::Matrix4d T_S_C = Eigen::Matrix4d::Identity();
    T_S_C.block<3,3>(0, 0) = q_S_C.toRotationMatrix();
    T_S_C.block<3,1>(0, 3) = p_S_C;
    T_S_Ci.push_back(T_S_C);

    sqrt_infos[i] = this->pixel_sigma_inverse_ * std::sqrt(
          camera_ptrs_[i]->fu());
  }

  // Triangulate the landmark from the observations
  Eigen::Vector3d l_W;
  TriangulationStatus status;
  if (!(*has_landmark_)) {
    status = svdTriangulation(
          T_W_Si, T_S_Ci, meas_normalized_, &l_W);
  } else {
    l_W = (*last_l_W_);
    status = nonlinearTriangulation(
          T_W_Si, T_S_Ci, meas_normalized_, sqrt_infos, &l_W,
          (*outliers_), 1.0);
  }

  if (status != TriangulationStatus::kSuccessful) {
    // Failed to extract landmark, hence, ignore this term
    Eigen::Map<Eigen::VectorXd> residual(residuals, num_residuals());
    residual.setZero();
    if (jacobians) {
      // Loop over all variables and compute the jacobian
      for (size_t i = 0; i < num_cams_; ++i) {
        const size_t jac_inc = i * parameter_count;
        // Jacobian w.r.t. T_W_S
        if (jacobians[jac_inc]) {
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
              Eigen::Dynamic, Eigen::RowMajor>>
              J(jacobians[jac_inc], num_residuals(),
                defs::visual::kPoseBlockSize);
          J.setZero();
        }

        // Jacobian w.r.t. T_S_C
        if (jacobians[jac_inc + kIdxCameraToImu]) {
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
              Eigen::Dynamic, Eigen::RowMajor>>
              J(jacobians[jac_inc + kIdxCameraToImu], num_residuals(),
              defs::visual::kPoseBlockSize);
          J.setZero();
        }
      }
    }

    return true;
  } else {
    (*last_l_W_) = l_W;
    (*has_landmark_) = true;

    // Check for outliers:
    size_t num_outliers = 0;
    for (size_t i = 0; i < num_cams_; ++i) {
      if ((*outliers_)[i]) {
        ++num_outliers;
      }
    }
  }

  // Compute the residual
  std::vector<bool> projection_success;
  projection_success.resize(num_cams_, true);
  Eigen::VectorXd residuals_raw(num_residuals() +
                                defs::visual::kPositionBlockSize);
  residuals_raw.setZero();
  std::vector<VisualJacobianType, Eigen::aligned_allocator<
      VisualJacobianType>> J_res_wrt_l_Wi;
  std::vector<PoseJacobianMinimal, Eigen::aligned_allocator<
      PoseJacobianMinimal>> J_res_wrt_T_W_Si;
  std::vector<PoseJacobianMinimal, Eigen::aligned_allocator<
      PoseJacobianMinimal>> J_res_wrt_T_S_Ci;


  J_res_wrt_l_Wi.resize(num_cams_, VisualJacobianType::Zero());
  if (jacobians) {
    J_res_wrt_T_W_Si.resize(num_cams_, PoseJacobianMinimal::Zero());
    J_res_wrt_T_S_Ci.resize(num_cams_, PoseJacobianMinimal::Zero());
  }

  // Variable to count the jacobian blocks
  size_t jacobian_blocks = 0;

  for (size_t i = 0; i < num_cams_; ++i) {
    // Project the triangulated landmark into frame
    Eigen::Vector2d projection;
    Eigen::Vector3d l_Si = q_W_Si[i].inverse() * (l_W - p_W_Si[i]);
    Eigen::Vector3d l_Ci = q_S_Ci[i].inverse() * (l_Si - p_S_Ci[i]);
    VisualJacobianType J_res_wrt_l_C;
    VisualJacobianType* J_res_wrt_l_C_ptr;
    if (jacobians) {
      J_res_wrt_l_C_ptr = &J_res_wrt_l_C;
    }

    aslam::ProjectionResult projection_result =
        camera_ptrs_[i]->project3(l_Ci, &projection, J_res_wrt_l_C_ptr);

    constexpr double kMaxDistanceFromOpticalAxisPxSquare = 1.0e5 * 1.0e5;
    constexpr double kMinDistanceToCameraPlane = 0.05;
    const bool projection_failed =
        (projection_result == aslam::ProjectionResult::POINT_BEHIND_CAMERA) ||
        (projection_result == aslam::ProjectionResult::PROJECTION_INVALID) ||
        (l_Ci(2, 0) < kMinDistanceToCameraPlane) ||
        (projection.squaredNorm() >
         kMaxDistanceFromOpticalAxisPxSquare);

    if (projection_failed) {
      projection_success[i] = false;
      continue;
    }

    // Success, compute the residual (this will be modified later)
    residuals_raw.segment<2>(i * 2) = (projection - measurements_[i]) *
        this->pixel_sigma_inverse_;

    // Collect the jacobians
    // Keep track of the index of the jacobian parameters
    const size_t jac_inc = i * parameter_count;

    // Prepare some jacobians
    J_res_wrt_l_Wi[i] = J_res_wrt_l_C * T_S_Ci[i].block<3,3>(0,0).transpose() *
        T_W_Si[i].block<3,3>(0, 0).transpose();

    if (jacobians) {
      if (jacobians[jac_inc]) {
        ++jacobian_blocks;
        if (projection_success[i] && !(*outliers_)[i]) {
          Eigen::Matrix<double, defs::visual::kPositionBlockSize, 6>
             J_l_C_wrt_T_W_S;
          J_l_C_wrt_T_W_S.setZero();
          J_l_C_wrt_T_W_S.block<3,3>(0,0) =
              T_S_Ci[i].block<3,3>(0, 0).transpose() *
              common::skew(l_Si); // Rotation
          J_l_C_wrt_T_W_S.block<3,3>(0,3) =
              -T_S_Ci[i].block<3,3>(0, 0).transpose() *
              T_W_Si[i].block<3,3>(0, 0).transpose(); // Translation
          J_res_wrt_T_W_Si[i] = J_res_wrt_l_C * J_l_C_wrt_T_W_S;
        }
      }
      if (jacobians[jac_inc + kIdxCameraToImu]) {
        ++jacobian_blocks;
        if (projection_success[i] && !(*outliers_)[i]) {
          Eigen::Matrix<double, defs::visual::kPositionBlockSize, 6>
              J_l_C_wrt_T_S_C;
          J_l_C_wrt_T_S_C.setZero();
          J_l_C_wrt_T_S_C.block<3,3>(0,0) = common::skew(l_Ci); // Rotation
          J_l_C_wrt_T_S_C.block<3,3>(0,3) =
              -T_S_Ci[i].block<3,3>(0, 0).transpose(); // Translation
          J_res_wrt_T_S_Ci[i] = J_res_wrt_l_C * J_l_C_wrt_T_S_C;
        }
      }
    }
  }

  // Put together the Jacobian matrix for the landmarks (Ja)
  Eigen::MatrixXd Ja(num_residuals() + defs::visual::kPositionBlockSize,
                     defs::visual::kPositionBlockSize);
  Ja.setZero();
  for (size_t i = 0; i < J_res_wrt_l_Wi.size(); ++i) {
    Ja.block(i * defs::visual::kResidualSize, 0,
              defs::visual::kResidualSize, defs::visual::kPositionBlockSize) =
        J_res_wrt_l_Wi[i];
  }

  // Decompose the Ja matrix
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_of_Ja(Ja, Eigen::ComputeFullU);
  Eigen::MatrixXd JbT = (svd_of_Ja.matrixU().block(0,
      defs::visual::kPositionBlockSize, num_residuals() + 3,
      num_residuals())).transpose();

  // Compute the actual residuals
  Eigen::Map<Eigen::VectorXd> residual(residuals, num_residuals());
  residual.setZero();
  residual = JbT * residuals_raw;


  // Compute the jacobians
  if (jacobians) {
    // Local paramterization to lift jacobian
    local_param::PoseQuaternionLocalParameterization local_param;
    Eigen::Matrix<double, 7, 6, Eigen::RowMajor> J_local_param;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        J_pose_min(num_residuals() + 3, defs::visual::kPoseBlockSize);

    // Loop over all variables and compute the jacobian
    for (size_t i = 0; i < num_cams_; ++i) {
      const size_t jac_inc = i * parameter_count;
      // Jacobian w.r.t. T_W_S
      if (jacobians[jac_inc]) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
            Eigen::Dynamic, Eigen::RowMajor>>
            J(jacobians[jac_inc], num_residuals(),
              defs::visual::kPoseBlockSize);
        if (projection_success[i]) {
          J_pose_min.setZero();
          local_param.ComputeJacobian(parameters[kIdxImuPose +
              jac_inc], J_local_param.data());
          J_local_param.block<4, 3>(0, 0) *= 4.0;
          J_pose_min.block<defs::visual::kResidualSize, 6>(
                jac_inc, 0) = J_res_wrt_T_W_Si[i];
          J = this->pixel_sigma_inverse_ * JbT *
              J_pose_min * J_local_param.transpose();

        } else {
          // Set the jacobian to zero
          J.setZero();
        }
      }

      // Jacobian w.r.t. T_S_C
      if (jacobians[jac_inc + kIdxCameraToImu]) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
            Eigen::Dynamic, Eigen::RowMajor>>
            J(jacobians[jac_inc + kIdxCameraToImu], num_residuals(),
            defs::visual::kPoseBlockSize);
        if (projection_success[i]) {
          J_pose_min.setZero();
          local_param.ComputeJacobian(parameters[kIdxCameraToImu +
              jac_inc], J_local_param.data());
          J_local_param.block<4, 3>(0, 0) *= 4.0;
          J_pose_min.block<defs::visual::kResidualSize, 6>(
                jac_inc, 0) = J_res_wrt_T_S_Ci[i];
          J = this->pixel_sigma_inverse_ * JbT *
              J_pose_min * J_local_param.transpose();
        } else {
          // Set the jacobian to zero
          J.setZero();
        }
      }
    }
  }

  return true;
}

} // namespace reprojection

} // namespace robopt
