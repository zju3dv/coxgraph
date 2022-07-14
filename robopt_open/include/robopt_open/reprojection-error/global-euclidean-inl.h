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
 * global-euclidean-inl.h
 * @brief Implementation file for reprojection residuals of landmarks to cameras 
 *        expressed in global, euclidean coordinates.
 * @author: Marco Karrer
 * Created on: Mar 19, 2018
 */

/// \brief robopt Main namespace of this package
namespace robopt {

namespace reprojection {

template <typename CameraType, typename DistortionType>
bool GlobalEuclideanReprError<CameraType, DistortionType>::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  // Coordinate frames:
  //  W = global
  //  S = IMU frame, expressed in W
  //  C = Camera frame, expressed in I

  // Unpack parameter blocks.
  Eigen::Map<const Eigen::Quaterniond> q_W_S(parameters[kIdxImuPose]);
  Eigen::Map<const Eigen::Vector3d> p_W_S(parameters[kIdxImuPose] +
      defs::visual::kOrientationBlockSize);
  Eigen::Map<const Eigen::Quaterniond> q_S_C(parameters[kIdxCameraToImu]);
  Eigen::Map<const Eigen::Vector3d> p_S_C(parameters[kIdxCameraToImu] + 
      defs::visual::kOrientationBlockSize);
  Eigen::Map<const Eigen::Vector3d> l_W(parameters[kIdxLandmark]);
  Eigen::Map<const Eigen::Matrix<double, CameraType::parameterCount(), 1>>
      intrinsics_map(parameters[kIdxCameraIntrinsics]);
  Eigen::Matrix<double, Eigen::Dynamic, 1> distortion_map;  
  if (DistortionType::parameterCount() > 0) {
    distortion_map = Eigen::Map<
        const Eigen::Matrix<double, DistortionType::parameterCount(), 1>>(
            parameters[kIdxCameraDistortion]); 
  }

  // Jacobian of landmark position in camera system w.r.t. keyframe pose
  Eigen::Matrix<double, defs::visual::kPositionBlockSize, 6> J_l_C_wrt_T_W_S;
  
  // Jacobian of landmark position in camera system w.r.t. IMU-to-cam transform
  Eigen::Matrix<double, defs::visual::kPositionBlockSize, 6> J_l_C_wrt_T_S_C;
  J_l_C_wrt_T_S_C.setZero();
  
  // Jacobian of landmark position in camera system w.r.t. landmark position
  Eigen::Matrix<double, defs::visual::kPositionBlockSize, 3> J_l_C_wrt_l_W;
  
  // Jacobians w.r.t. camera intrinsics and distortion coefficients
  typedef Eigen::Matrix<double, defs::visual::kResidualSize, Eigen::Dynamic>
      JacobianWrtIntrinsicsType;
  JacobianWrtIntrinsicsType J_keypoint_wrt_intrinsics(
      defs::visual::kResidualSize, CameraType::parameterCount());
  typedef Eigen::Matrix<double, defs::visual::kResidualSize, Eigen::Dynamic>
      JacobianWrtDistortionType;
  JacobianWrtDistortionType J_keypoint_wrt_distortion(
      defs::visual::kResidualSize, DistortionType::parameterCount());

  // Extract the rotation matrices
  const Eigen::Matrix3d R_W_S = q_W_S.toRotationMatrix();
  const Eigen::Matrix3d R_S_C = q_S_C.toRotationMatrix(); 
  
  // Transform Landmark into camera coordinates
  const Eigen::Vector3d l_S = R_W_S.transpose()*(l_W - p_W_S);
  const Eigen::Vector3d l_C = R_S_C.transpose()*(l_S - p_S_C);
  
  // Jacobian of 2d keypoint (including distortion and intrinsics)
  // w.r.t. to landmark position in camera coordinates
  Eigen::Vector2d reprojected_landmark;
  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPositionBlockSize>
      VisualJacobianType;

  VisualJacobianType J_keypoint_wrt_l_C;
  Eigen::VectorXd intrinsics = intrinsics_map;
  Eigen::VectorXd distortion = distortion_map;
  
  // Only evaluate the jacobian if requested.
  VisualJacobianType* J_keypoint_wrt_l_C_ptr = nullptr;
  if (jacobians) {
    J_keypoint_wrt_l_C_ptr = &J_keypoint_wrt_l_C;
  }
  JacobianWrtIntrinsicsType* J_keypoint_wrt_intrinsics_ptr = nullptr;
  if (jacobians && jacobians[kIdxCameraIntrinsics]) {
    J_keypoint_wrt_intrinsics_ptr = &J_keypoint_wrt_intrinsics;
  }
  JacobianWrtDistortionType* J_keypoint_wrt_distortion_ptr = nullptr;
  if (DistortionType::parameterCount() > 0 && jacobians &&
      jacobians[kIdxCameraDistortion]) {
    J_keypoint_wrt_distortion_ptr = &J_keypoint_wrt_distortion;
  }

  const aslam::ProjectionResult projection_result =
      camera_ptr_->project3Functional(
          l_C, &intrinsics, &distortion, &reprojected_landmark,
          J_keypoint_wrt_l_C_ptr, J_keypoint_wrt_intrinsics_ptr,
          J_keypoint_wrt_distortion_ptr);
  
  // Handle projection failures by zeroing the Jacobians and setting the error
  // to zero.
  constexpr double kMaxDistanceFromOpticalAxisPxSquare = 1.0e5 * 1.0e5;
  constexpr double kMinDistanceToCameraPlane = 0.05;
  const bool projection_failed =
      (projection_result == aslam::ProjectionResult::POINT_BEHIND_CAMERA) ||
      (projection_result == aslam::ProjectionResult::PROJECTION_INVALID) ||
      (l_C(2, 0) < kMinDistanceToCameraPlane) ||
      (reprojected_landmark.squaredNorm() >
       kMaxDistanceFromOpticalAxisPxSquare);

  if (jacobians) {
    if (projection_failed && J_keypoint_wrt_intrinsics_ptr != nullptr) {
      J_keypoint_wrt_intrinsics_ptr->setZero();
    }
    if (projection_failed && J_keypoint_wrt_intrinsics_ptr != nullptr) {
      J_keypoint_wrt_distortion_ptr->setZero();
    }
  
    // JPL quaternion parameterization is used because our memory layout
    // of quaternions is JPL.
    local_param::QuaternionLocalParameterization quat_parameterization;

    // These Jacobians will be used in all the cases.
    J_l_C_wrt_T_S_C.block<3,3>(0,0) = common::skew(l_C); // Rotation
    J_l_C_wrt_T_S_C.block<3,3>(0,3) = -R_S_C.transpose(); // Translation
  
    J_l_C_wrt_T_W_S.block<3,3>(0,0) = R_S_C.transpose()*
        common::skew(l_S); // Rotation
    J_l_C_wrt_T_W_S.block<3,3>(0,3) = -R_S_C.transpose()*
        R_W_S.transpose(); // Translation

    // Jacobian w.r.t. global landmark mission base pose.
    if (jacobians[kIdxImuPose]) {
      Eigen::Map<PoseJacobian> J(jacobians[kIdxImuPose]);
      if (!projection_failed) {
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
        quat_parameterization.ComputeJacobian(
            q_W_S.coeffs().data(), J_quat_local_param.data());

        J.leftCols(defs::visual::kOrientationBlockSize) =
            J_keypoint_wrt_l_C * J_l_C_wrt_T_W_S.block<3,3>(0,0) * 4.0 *
            J_quat_local_param.transpose() * this->pixel_sigma_inverse_;
        J.rightCols(defs::visual::kPositionBlockSize) = 
            J_keypoint_wrt_l_C * J_l_C_wrt_T_W_S.block<3,3>(0,3) *
            this->pixel_sigma_inverse_;
      } else {
        J.setZero();
      }
    }

    // Jacobian w.r.t. camera-to-IMU
    if (jacobians[kIdxCameraToImu]) {
      Eigen::Map<PoseJacobian> J(jacobians[kIdxCameraToImu]);
      if (!projection_failed) {
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
        quat_parameterization.ComputeJacobian(
            q_S_C.coeffs().data(), J_quat_local_param.data());
        J.leftCols(defs::visual::kOrientationBlockSize) = 
            J_keypoint_wrt_l_C * J_l_C_wrt_T_S_C.block<3,3>(0,0) * 4.0 *
            J_quat_local_param.transpose() * this->pixel_sigma_inverse_;
        J.rightCols(defs::visual::kPositionBlockSize) = 
            J_keypoint_wrt_l_C * J_l_C_wrt_T_S_C.block<3,3>(0,3) *
            this->pixel_sigma_inverse_;
      } else {
        J.setZero();
      }
    }

    // Jacobian w.r.t. landmark
    if (jacobians[kIdxLandmark]) {
      Eigen::Map<PositionJacobian> J(jacobians[kIdxLandmark]);
      if (!projection_failed) {
        J_l_C_wrt_l_W = R_S_C.transpose()*R_W_S.transpose();
        J = J_keypoint_wrt_l_C * J_l_C_wrt_l_W * this->pixel_sigma_inverse_;
      } else {
        J.setZero();
      }
    }

    // Jacobian w.r.t. intrinsics.
    if (jacobians[kIdxCameraIntrinsics]) {
      Eigen::Map<IntrinsicsJacobian> J(jacobians[kIdxCameraIntrinsics]);
      if (!projection_failed) {
        J = J_keypoint_wrt_intrinsics * this->pixel_sigma_inverse_;
      } else {
        J.setZero();
      }
    }

    // Jacobian w.r.t. distortion.
    if (DistortionType::parameterCount() > 0 &&
        jacobians[kIdxCameraDistortion]) {
      Eigen::Map<DistortionJacobian> J(
          jacobians[kIdxCameraDistortion], defs::visual::kResidualSize,
          DistortionType::parameterCount());
      if (!projection_failed) {
        J = J_keypoint_wrt_distortion * this->pixel_sigma_inverse_;
      } else {
        J.setZero();
      }
    }
  }

  // Compute residuals.
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  if (!projection_failed) {
    residual =
        (reprojected_landmark - measurement_) * this->pixel_sigma_inverse_;
  } else {
    residual.setZero();
  }
  
  return true;
}

} // namespace reprojection

} // namespace robopt
