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
 * map.hpp
 * @brief Header for the Map class.
 * @author: Marco Karrer
 * Created on: Aug 15, 2018
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "parameters.hpp"
#include "pose_graph_backend/keyframe.hpp"

/// \brief pgbe The main namespace of this package
namespace pgbe {

class Map {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::list<std::shared_ptr<KeyFrame>,
                    Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
      KFlist;
  typedef std::vector<std::shared_ptr<KeyFrame>,
                      Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
      KFvec;
  typedef std::set<std::shared_ptr<KeyFrame>,
                   std::less<std::shared_ptr<KeyFrame>>,
                   Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
      KFset;

  /// \brief Constructor
  /// @param params The system parameters.
  /// @param agent_id The id of the agent for this map.
  Map(const SystemParameters& parameters, const uint64_t agent_id);

  ~Map(){};

  /// \brief Get the agent id for this map.
  /// @return The agent id.
  uint64_t getAgentId() { return agent_id_; }

  /// \brief Add a new keyframe.
  /// @param kf_ptr The keyframe pointer.
  /// @return Whether or not the keyframe could be added.
  bool addKeyFrame(std::shared_ptr<KeyFrame> kf_ptr);

  /// \brief Get a keyframe.
  /// @param id The keyframe identifier.
  std::shared_ptr<KeyFrame> getKeyFrame(const Identifier& id);

  /// \brief Get all keyframes.
  /// @return A vector containing all keyframes.
  KFvec getAllKeyFrames();

  /// \brief Get the most recent N keyframes from this map.
  /// @param N The number of keyframes.
  /// @return A vector containing the N (or the max number of KFs in the map)
  ///          most recent KFs
  KFvec getMostRecentN(const int& N);

  /// \brief Set the local odometry frame pose for this map.
  /// @param T_M_O The transformation from the odometry to the map frame.
  void setOdomToMap(const Eigen::Matrix4d& T_M_O) {
    std::lock_guard<std::mutex> lock(mutex_);
    T_M_O_ = T_M_O;
  }

  /// \brief Get the local odometry frame pose for this map.
  /// @return T_M_O The transformation.
  Eigen::Matrix4d getOdomToMap() { return T_M_O_; }

  /// \brief Set the map to world frame transforamtion.
  /// @param T_W_M The transformation.
  /// @param covariance The covariance of the transformation
  void setWorldTransformation(const Eigen::Matrix4d& T_W_M,
                              const Eigen::Matrix4d& covariance);

  /// \brief Check whether there exists a valid world to reference trans.
  /// @return Whether or not there is a valid transformation.
  bool hasValidWorldTransformation() { return has_init_T_W_M_; }

  /// \brief Get the map to world frame transformation.
  /// @return The tranformation T_W_M.
  Eigen::Matrix4d getWorldTransformation() { return T_W_M_; }

  /// \brief Get the map to world frame transformation covariance.
  /// @return The tranformation T_R_W covariance.
  Eigen::Matrix4d getWorldTransformationCov() { return cov_T_W_M_; }

  /// \brief Get the GPS antenna position in the IMU frame.
  /// @return The antenna position.
  Eigen::Vector3d getGpsAntennaPosition() {
    return parameters_.gps_parameters[agent_id_].offset;
  }

  /// \brief Add other agent id's to collection of maps that
  /// have been merged thanks to loop closure between them
  /// @param merged_id The other agent's id
  void addMergedAgents(uint64_t merged_id);

  /// \brief Get the merged agents for this map
  std::vector<uint64_t> getMergedAgents() { return merged_agents_; }

  /// \brief Write the keyframe poses to a csv file.
  /// @param filename The filename.
  void writePosesToFile(const std::string& filename);

  void writeOdomPosesToFile(const std::string& filename);

  /// \brief Write the world transformed keyframe poses to a csv file.
  /// @param filename The filename.
  void writePosesToFileInWorld(const std::string& filename);

  /// \brief Return whether the agent has active GPS
  bool getGPSStatus() { return gps_active_; }

  /// \brief Set whether the agent has active GPS
  /// @param gps_active If agent has active GPS
  void setGPSStatus(bool gps_active) { gps_active_ = gps_active; }

  /// \brief Set current agent as a world frame anchor
  /// for agents with no GPS
  /// @param gps_active If agent has active GPS
  void setWorldAnchor() { world_anchor_ = true; }

  /// \brief Get if current agent is a world frame anchor
  bool getWorldAnchor() { return world_anchor_; }

  /// \brief Store the latest loop closure info as merged agent and transform
  /// @param merged_agent_id The agent a loop was just closed with
  /// @param T_A_B The relative pose transformation from loop closure
  void setNewMerge(Identifier merged_agent_id, Eigen::Matrix4d T_A_B) {
    last_merge_ = std::make_pair(merged_agent_id, T_A_B);
  }

  /// \brief Get the latest loop closure info
  std::pair<Identifier, Eigen::Matrix4d> getNewMerge() { return last_merge_; }

  /// \brief Get the size of the map
  int getMapSize() { return keyframe_id_map_.size(); }

 protected:
  // Sorting function for keyframes
  static bool keyframeIdSort(std::shared_ptr<KeyFrame> kf_i,
                             std::shared_ptr<KeyFrame> kf_j) {
    return (kf_i->getId().second < kf_j->getId().second);
  }

  // The system parameters
  SystemParameters parameters_;

  // Store the agent-id for this map
  const uint64_t agent_id_;

  // Store the local odometry transformation(s)
  Eigen::Matrix4d T_M_O_;

  // Store the transformation from world to the reference frame (GPS).
  Eigen::Matrix4d T_W_M_;
  Eigen::Matrix4d cov_T_W_M_;
  bool has_init_T_W_M_;

  // Store the Keyframe data
  std::map<Identifier, std::shared_ptr<KeyFrame>> keyframe_id_map_;

  // Lock for map access.
  std::mutex mutex_;

  // Store which other maps have been merged with current map
  std::vector<uint64_t> merged_agents_;

  // If agent has active GPS
  bool gps_active_;

  // If agent is world frame anchor for other agents without GPS
  bool world_anchor_;

  // Latest loop detection information, with agent id and transformation
  std::pair<Identifier, Eigen::Matrix4d> last_merge_;
};

}  // namespace pgbe
