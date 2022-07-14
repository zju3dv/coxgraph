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
 * typedefs.hpp
 * @brief Useful typedefs.
 * @author: Marco Karrer
 * Created on: Aug 13, 2018
 */

#pragma once

#include <Eigen/Dense>
#include <deque>
#include <vector>

#include <sensor_msgs/PointCloud2.h>

/// \brief pgbe Main namespace of this package.
namespace pgbe {

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
    Vector2Vector;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
    Vector3Vector;
typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
    Vector4Vector;

typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
    Matrix4Vector;

typedef std::pair<uint64_t, uint64_t> Identifier;  // First:=Agent,
                                                   // Second:=Frame

/// @brief Type to store the result of matching.
struct Match {
  /// @brief Constructor.
  /// @param idxA_ Keypoint index of frame A.
  /// @param idxB_ Keypoint index of frame B.
  /// @param distance_ Descriptor distance between those two keypoints.
  Match(size_t idx_A_, size_t idx_B_, float distance_)
      : idx_A(idx_A_), idx_B(idx_B_), distance(distance_) {}
  size_t idx_A;
  size_t idx_B;
  float distance;
};

typedef std::vector<Match> Matches;

/// @brief Type to store a loop edge.
struct LoopEdge {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// @brief Constructor.
  /// @param id_A_ The keyframe identifier for keyframe A.
  /// @param id_B_ The keyframe identifier for keyframe B.
  /// @param T_A_B_ The relative transformation from keyframe B to A
  ///         (their corresponding IMU frame).
  LoopEdge(const Identifier& id_A_, const Identifier& id_B_,
           const Eigen::Matrix4d& T_A_B_)
      : id_A(id_A_), id_B(id_B_), T_A_B(T_A_B_) {}
  Identifier id_A;
  Identifier id_B;
  Eigen::Matrix4d T_A_B;
};

typedef std::vector<LoopEdge, Eigen::aligned_allocator<LoopEdge>> LoopEdges;

/// @brief Struct to store data for publishing
struct Result {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Result()
      : timestamp(0.0),
        T_M_O(Eigen::Matrix4d::Identity()),
        T_W_M(Eigen::Matrix4d::Identity()),
        T_M_Si(Eigen::Matrix4d::Identity()) {}
  Result(const double timestamp_, const Eigen::Matrix4d& T_M_O_,
         const Eigen::Matrix4d& T_W_M_, const Eigen::Matrix4d& T_M_Si_)
      : timestamp(timestamp_), T_M_O(T_M_O_), T_W_M(T_W_M_), T_M_Si(T_M_Si_) {}

  // Data that is always available
  double timestamp;       // The timestamp of the keyframe.
  Eigen::Matrix4d T_M_O;  // Transformation odometry to map.
  Eigen::Matrix4d T_W_M;  // Transformation map to world (GPS).
  Eigen::Matrix4d T_M_Si;

  // Data only available upon loop closure.
  Matrix4Vector T_W_Cs;  // The transformations camera to world.
  std::vector<sensor_msgs::PointCloud2> point_clouds;
};

}  // namespace pgbe
