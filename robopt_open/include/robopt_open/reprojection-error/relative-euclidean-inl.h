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
 * relative-euclidean-inl.h
 * @brief Implementation file for reprojection residuals of fixed landmarks
 *    expressed in a relative frame to a camera.
 * @author: Marco Karrer
 * Created on: Aug 17, 2018
 */

/// \brief robopt Main namespace of this package
namespace robopt {

namespace reprojection {

template<typename CameraType, typename DistortionType>
bool RelativeEuclideanReprError<CameraType, DistortionType>::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  // Coordinate frames.
  // A, B = defining the relative frames (parameters is assumed to give T_A_B)
  // C = camera (where the projection is performed)
  // R = reference frame (where the landmarks are expressed.

  // Unpack parameter blocks. Note: The optimized pose is alway
  Eigen::Map<const Eigen::Quaterniond> q_A_B(parameters[0]);
  Eigen::Map<const Eigen::Vector3d> p_A_B(parameters[0] +
      defs::visual::kOrientationBlockSize);

  // Compose the transformation to transform the point into the camera frame.
  Eigen::Matrix3d R_C_R;
  Eigen::Vector3d p_C_R;
  if (error_term_type_ == defs::visual::RelativeProjectionType::kNormal) {
    R_C_R = q_A_B.toRotationMatrix();
    p_C_R = p_A_B;
  } else if (error_term_type_ ==
             defs::visual::RelativeProjectionType::kInverse){
    R_C_R = (q_A_B.inverse()).toRotationMatrix();
    p_C_R = - (q_A_B.inverse() * p_A_B);
  } else {
    // Wrong type --> fail the optimization!
    return false;
  }

  // Transform the point into the camera frame
  const Eigen::Vector3d l_C = R_C_R * point_ref_ + p_C_R;
  typedef Eigen::Matrix<double, defs::visual::kResidualSize,
                        defs::visual::kPositionBlockSize>
      VisualJacobianType;
  VisualJacobianType J_keypoint_wrt_l_C;

  // Only evaluate the jacobian if requested.
  VisualJacobianType* J_keypoint_wrt_l_C_ptr = nullptr;
  if (jacobians) {
    J_keypoint_wrt_l_C_ptr = &J_keypoint_wrt_l_C;
  }

  // Evaluate the residual
  Eigen::Vector2d reprojection;
  const aslam::ProjectionResult projection_result =
      camera_ptr_->project3(l_C, &reprojection, J_keypoint_wrt_l_C_ptr);

  // Handle projection failures by zeroing the Jacobians and setting the error
  // to zero.
  constexpr double kMaxDistanceFromOpticalAxisPxSquare = 1.0e5 * 1.0e5;
  constexpr double kMinDistanceToCameraPlane = 0.05;
  const bool projection_failed =
      (projection_result == aslam::ProjectionResult::POINT_BEHIND_CAMERA) ||
      (projection_result == aslam::ProjectionResult::PROJECTION_INVALID) ||
      (l_C(2, 0) < kMinDistanceToCameraPlane) ||
      (reprojection.squaredNorm() >
       kMaxDistanceFromOpticalAxisPxSquare);

  // Compute the jacobians
  if (jacobians) {
    // JPL quaternion parameterization is used because our memory layout
    // of quaternions is JPL.
    local_param::QuaternionLocalParameterization quat_parameterization;

    // These Jacobians will be used in all the cases.
    Eigen::Matrix<double, defs::visual::kPositionBlockSize, 6> J_l_C_wrt_T_C_R;
    J_l_C_wrt_T_C_R.setZero();
    J_l_C_wrt_T_C_R.block<3,3>(0,0) = -R_C_R * common::skew(point_ref_); // Rot.
    J_l_C_wrt_T_C_R.block<3,3>(0,3) = Eigen::Matrix3d::Identity(); // Transl.
    Eigen::Matrix<double, 6, 6> J_T_C_R_wrt_T_A_B;
    if (error_term_type_ == defs::visual::RelativeProjectionType::kNormal) {
      J_T_C_R_wrt_T_A_B.setIdentity();
    } else { // Allready did a check for valid type before
      Eigen::Matrix3d R_A_B = q_A_B.toRotationMatrix();
      J_T_C_R_wrt_T_A_B.setIdentity();
      J_T_C_R_wrt_T_A_B.block<3, 3>(0,0) = -R_A_B;
      J_T_C_R_wrt_T_A_B.block<3, 3>(3,0) = common::skew(p_C_R);
      J_T_C_R_wrt_T_A_B.block<3, 3>(3,3) = -R_A_B.transpose();
    }

    if (jacobians[0]) {
      Eigen::Map<PoseJacobian> J(jacobians[0]);
      if (!projection_failed) {
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
        quat_parameterization.ComputeJacobian(
              q_A_B.coeffs().data(), J_quat_local_param.data());
        Eigen::Matrix<double, 6, 7> J_local;
        J_local.setZero();
        J_local.topLeftCorner(3, 4) = J_quat_local_param.transpose() * 4.0;
        J_local.bottomRightCorner(3,3) = Eigen::Matrix3d::Identity();
        J = J_keypoint_wrt_l_C * J_l_C_wrt_T_C_R * J_T_C_R_wrt_T_A_B *
            J_local * this->pixel_sigma_inverse_;
      } else {
        J.setZero();
      }
    }
  }

  // Compute residuals.
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  if (!projection_failed) {
    residual =
        (reprojection - measurement_) * this->pixel_sigma_inverse_;
  } else {
    residual.setZero();
  }

  return true;
}

} // namespace reprojection

} // namespace robopt
