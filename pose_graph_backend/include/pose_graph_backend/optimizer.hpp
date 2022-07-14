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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * optimizer.hpp
 * @brief Header file for the Optimizer Class
 * @author: Marco Karrer
 * Created on: Aug 17, 2018
 */

#pragma once

#include <ceres/ceres.h>

#include "pose_graph_backend/keyframe.hpp"
#include "pose_graph_backend/map.hpp"
#include "typedefs.hpp"

/// \brief pgbe The main namespace of this package
namespace pgbe {

// Callback to enable
class CeresStoppCallback : public ceres::IterationCallback {
 public:
  CeresStoppCallback(bool* stop_flag) { stop_flag = stop_flag; }

  ~CeresStoppCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    if (*stop_flag) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  bool* stop_flag;
};

class Optimizer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::vector<std::shared_ptr<Map>,
                      Eigen::aligned_allocator<std::shared_ptr<Map>>>
      MapVec;

  /// \brief Convert and write a pose to a ceres parameter pointer.
  /// @param ceres_ptr The pointer to the data used by ceres.
  /// @param T  The homogeneous transformation matrix.
  static void homogenousToCeres(const Eigen::Matrix4d& T, double* ceres_ptr);

  /// \brief Convert ceres parameter pointer to a homogeneous transform.
  /// @param ceres_ptr The pointer to the data used by ceres.
  /// @return The homogeneous transformation matrix.
  static Eigen::Matrix4d ceresToHomogenous(double* ceres_ptr);

  /// \brief Compute initial alignment for GPS
  /// @param correspondences The GPS-Odometry correspondences.
  /// @param antenna_pos The position of the GPS antenna in IMU coordinates.
  /// @param T_W_O The transforamtion from the odometry origin to world.
  /// @param covariance The covariance of the transforamtion.
  /// @return Whether or not the operation was successful.
  static bool computeGPSalignment(const OdomGPScombinedVector& correspondences,
                                  const Eigen::Vector3d& antenna_pos,
                                  Eigen::Matrix4d& T_W_O,
                                  Eigen::Matrix4d& covariance,
                                  SystemParameters& params,
                                  OdomGPScombinedQueue& odom_gps_init_queue);

  /// \brief Compute the transformation between odom and map frame.
  /// @param keyframes A vector containing the keyframes that should be used
  ///          to compute the alignment (they must be from the same agent!).
  /// @param T_M_O_init The initial guess for the transformation.
  /// @return The homegenous transformation (T_World_Odom)
  static Eigen::Matrix4d computeMapTransformation(
      const Map::KFvec& keyframes, const Eigen::Matrix4d& T_M_O_init,
      SystemParameters& system_parameters);

  /// \brief Optimize the relative pose from a loop-closure correspondence.
  /// @param keyframe_A The first keyframe.
  /// @param keyframe_B The second keyframe.
  /// @param landmarks_from_A_in_B Matched landmarks expressed in Keyfram A
  ///         found in Keyframe B.
  /// @param landmarks_from_B_in_A Matched landmarks expressed in Keyframe B
  ///         found in Keyframe B.
  /// @param T_A_B The relative pose from Keyframe B to Keyframe A.
  /// @param th2 The outlier rejection threshold.
  /// @return The number of inlier correspondences.
  static int optimizeRelativePose(std::shared_ptr<KeyFrame> keyframe_A,
                                  std::shared_ptr<KeyFrame> keyframe_B,
                                  const Matches& landmarks_from_B_in_A,
                                  const Matches& landmarks_from_A_in_B,
                                  Eigen::Matrix4d& T_A_B,
                                  SystemParameters& params);

  /// \brief Perform a full pose graph optimization of a map.
  /// @param maps_ptr Pointer(s) to the map(s) which will be optimized.
  /// @param stop_flag Signal other optimizations that the result from the
  ///         global optimization is written to the map.
  static void optimizeMapPoseGraph(const MapVec& maps_ptr, bool* stop_flag,
                                   SystemParameters& system_parameters);

  /// \brief Optimize local window.
  /// @param map_ptr A pointer to the map which will be optimized.
  /// @param window_size The number of keyframes that should be in the window.
  /// @param stop_flag Flag to interrupt optimization when a global opt. has
  ///        finished in the meanwhile.
  static void optimizeLocalPoseGraph(std::shared_ptr<Map> map_ptr,
                                     const size_t& window_size, bool* stop_flag,
                                     SystemParameters& system_parameters);
};

}  // namespace pgbe
