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
 * @brief Source for the Map class.
 * @author: Marco Karrer
 * Created on: Aug 15, 2018
 */

#include "pose_graph_backend/map.hpp"

namespace pgbe {

Map::Map(const SystemParameters& params, const uint64_t agent_id)
    : parameters_(params),
      agent_id_(agent_id),
      has_init_T_W_M_(false),
      T_M_O_(Eigen::Matrix4d::Identity()) {
  merged_agents_.push_back(agent_id);
  world_anchor_ = false;
}

bool Map::addKeyFrame(std::shared_ptr<KeyFrame> kf_ptr) {
  Identifier kf_id = kf_ptr->getId();

  // Check if the keyframe is actually from the same agent.
  if (kf_id.first != agent_id_) {
    return false;
  }

  KFvec connected_kfs;
  std::set<Identifier> connections_odom_new;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (keyframe_id_map_.count(kf_id)) {
      return false;
    }

    keyframe_id_map_.insert(std::make_pair(kf_id, kf_ptr));

    // Get the connected keyframes.
    connections_odom_new = kf_ptr->getOdomConnections();
    for (auto itr = connections_odom_new.begin();
         itr != connections_odom_new.end(); ++itr) {
      if (keyframe_id_map_.count((*itr))) {
        connected_kfs.push_back(keyframe_id_map_[(*itr)]);
      }
    }
  }

  // Search for the missing connection
  for (auto itr = connected_kfs.begin(); itr != connected_kfs.end(); ++itr) {
    std::set<Identifier> old_connections = (*itr)->getOdomConnections();
    for (auto itr2 = connections_odom_new.begin();
         itr2 != connections_odom_new.end(); ++itr2) {
      if (!old_connections.count((*itr2))) {
        (*itr)->insertOdomConnection(itr2->second);
      }
    }
  }
  return true;
}

std::shared_ptr<KeyFrame> Map::getKeyFrame(const Identifier& id) {
  std::shared_ptr<KeyFrame> requested_kf(NULL);
  if (id.first != agent_id_) {
    return requested_kf;
    std::cout << "ERROR: getKeyFrame agent_id's don't match" << std::endl;
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (keyframe_id_map_.count(id)) {
      requested_kf = keyframe_id_map_[id];
    }
  }

  return requested_kf;
}

std::vector<std::shared_ptr<KeyFrame>,
            Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
Map::getAllKeyFrames() {
  std::unique_lock<std::mutex> lock(mutex_);

  KFvec keyframes;
  keyframes.reserve(keyframe_id_map_.size());
  for (auto itr = keyframe_id_map_.begin(); itr != keyframe_id_map_.end();
       ++itr) {
    keyframes.push_back(itr->second);
  }

  return keyframes;
}

std::vector<std::shared_ptr<KeyFrame>,
            Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
Map::getMostRecentN(const int& N) {
  std::unique_lock<std::mutex> lock(mutex_);

  KFvec keyframes;
  if (keyframe_id_map_.size() == 0) {
    return keyframes;
  }

  keyframes.reserve(keyframe_id_map_.size());
  for (auto itr = keyframe_id_map_.begin(); itr != keyframe_id_map_.end();
       ++itr) {
    if (itr->first.first == agent_id_) {
      keyframes.push_back(itr->second);
    }
  }

  std::sort(keyframes.begin(), keyframes.end(), this->keyframeIdSort);

  KFvec recent_keyframes;
  recent_keyframes.reserve(N);
  int num_kfs = N;
  if (keyframes.size() < N) {
    num_kfs = keyframes.size();
  }

  for (size_t i = keyframes.size() - 1; i > keyframes.size() - num_kfs; --i) {
    recent_keyframes.push_back(keyframes[i]);
  }

  return recent_keyframes;
}

void Map::setWorldTransformation(const Eigen::Matrix4d& T_W_M,
                                 const Eigen::Matrix4d& covariance) {
  std::lock_guard<std::mutex> lock(mutex_);
  has_init_T_W_M_ = true;
  T_W_M_ = T_W_M;
  cov_T_W_M_ = covariance;
}

void Map::writePosesToFile(const std::string& filename) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream file;
  file.open(filename);
  const Eigen::Quaterniond q_W_M(T_W_M_.block<3, 3>(0, 0));
  const Eigen::Vector3d p_W_M = T_W_M_.block<3, 1>(0, 3);

  file << q_W_M.w() << "," << q_W_M.x() << "," << q_W_M.y() << ",";
  file << q_W_M.z() << '\n';
  for (auto itr = keyframe_id_map_.begin(); itr != keyframe_id_map_.end();
       ++itr) {
    std::shared_ptr<KeyFrame> keyframe_i = itr->second;
    const double timestamp = keyframe_i->getTimestamp();
    const Eigen::Matrix4d T_M_Si = keyframe_i->getOptimizedPose();
    const Eigen::Quaterniond q_M_Si(T_M_Si.block<3, 3>(0, 0));
    const Eigen::Vector3d p_M_Si = T_M_Si.block<3, 1>(0, 3);
    file << std::setprecision(25) << timestamp << ",";
    file << p_M_Si(0) << "," << p_M_Si(1) << "," << p_M_Si(2) << ",";
    file << q_M_Si.w() << "," << q_M_Si.x() << "," << q_M_Si.y() << ",";
    file << q_M_Si.z() << '\n';
  }
  file.close();
}

void Map::writeOdomPosesToFile(const std::string& filename) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream file;
  file.open(filename, std::ios_base::app);
  const Eigen::Quaterniond q_M_O(T_M_O_.block<3, 3>(0, 0));
  const Eigen::Vector3d p_M_O = T_M_O_.block<3, 1>(0, 3);

  file << std::setprecision(25);
  file << p_M_O(0) << "," << p_M_O(1) << "," << p_M_O(2) << ",";
  file << q_M_O.w() << "," << q_M_O.x() << "," << q_M_O.y() << ",";
  file << q_M_O.z() << std::endl;
  file.close();
}

void Map::writePosesToFileInWorld(const std::string& filename) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream file;
  file.open(filename);
  const Eigen::Quaterniond q_W_M(T_W_M_.block<3, 3>(0, 0));
  const Eigen::Vector3d p_W_M = T_W_M_.block<3, 1>(0, 3);

  const Eigen::Quaterniond q_M_O(T_M_O_.block<3, 3>(0, 0));
  const Eigen::Vector3d p_M_O = T_M_O_.block<3, 1>(0, 3);
  file << p_W_M(0) << "," << p_W_M(1) << "," << p_W_M(2) << "," << q_W_M.w()
       << "," << q_W_M.x() << "," << q_W_M.y() << "," << q_W_M.z() << std::endl;
  file << p_M_O(0) << "," << p_M_O(1) << "," << p_M_O(2) << "," << q_M_O.w()
       << "," << q_M_O.x() << "," << q_M_O.y() << "," << q_M_O.z() << std::endl;
  for (auto itr = keyframe_id_map_.begin(); itr != keyframe_id_map_.end();
       ++itr) {
    std::shared_ptr<KeyFrame> keyframe_i = itr->second;
    const double timestamp = keyframe_i->getTimestamp();
    const Eigen::Matrix4d T_M_Si = keyframe_i->getOptimizedPose();
    const Eigen::Matrix4d T_W_Si = T_W_M_ * T_M_Si;

    const Eigen::Quaterniond q_W_Si(T_W_Si.block<3, 3>(0, 0));
    const Eigen::Vector3d p_W_Si = T_W_Si.block<3, 1>(0, 3);
    file << std::setprecision(25);
    file << timestamp << ",";
    file << p_W_Si(0) << "," << p_W_Si(1) << "," << p_W_Si(2) << ",";
    file << q_W_Si.w() << "," << q_W_Si.x() << "," << q_W_Si.y() << ",";
    file << q_W_Si.z() << '\n';
  }
  file.close();
}

void Map::addMergedAgents(uint64_t merged_id) {
  if (std::find(merged_agents_.begin(), merged_agents_.end(), merged_id) ==
      merged_agents_.end()) {
    merged_agents_.push_back(merged_id);
  }
}

}  // namespace pgbe
