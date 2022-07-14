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
 * triangulation.h
 * @brief Header file for triangulation functionality.
 *        (Note: This functionality is mostly copied/adapted from aslam_cv2)
 * @author: Marco Karrer
 * Created on: Nov 15, 2018
 */

#pragma once

#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

/// \brief robopt The main namespace of this package.
namespace robopt {

namespace reprojection {

/// Possible triangulation state.
enum class TriangulationStatus {
  /// The triangulation was successful.
  kSuccessful,
  /// There were too few landmark observations.
  kTooFewMeasurments,
  /// The landmark is not fully observable (rank deficiency).
  kUnobservable
};

/// \brief Simple SVD based triangulation.
/// @param T_W_Si Vector containing the T_W_S (as homogeneous transformations).
/// @param T_S_Ci Vector containing the T_S_C (as homogeneous transformations).
/// @param meas_normalized The image point measurments, undistorted and in
///         normalized image coordiantes.
/// @param l_W The triangulated landmark in W.
/// @return The triangulation status.
TriangulationStatus svdTriangulation(
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
      T_W_Si,
  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
      T_S_Ci,
  const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
      meas_normalized,
  Eigen::Vector3d* l_W);

/// \brief Nonlinear triangulation refinement.
/// @param T_W_Si Vector containing the T_W_S (as homogeneous transformations).
/// @param T_S_Ci Vector containing the T_S_C (as homogeneous transformations).
/// @param meas_normalized The image point measurments, undistorted and in
///         normalized image coordiantes.
/// @param sqrt_info The scalar square root information of the keypoints in
///         meas_normalized (Note: this has to be in normalized coordinates!)
/// @param l_W The triangulated landmark in W. This function needs an initial
///         guess for l_W (e.g. by svdTriangulation).
/// @param outliers Vector containing whether a point was classified as outlier
///         (--> outlier = true) or not.
/// @param c The cauchy length factor.
/// @return The triangulation status.
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
  const double c = 1.0);

} // namespace reprojection

} // namespace robopt
