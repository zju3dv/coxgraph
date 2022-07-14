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
 * keyframe.cpp
 * @brief Source file for the KeyFrame Class
 * @author: Marco Karrer
 * Created on: Aug 14, 2018
 */

#include <cv_bridge/cv_bridge.h>
#include <algorithm>

#include "pose_graph_backend/keyframe.hpp"

namespace pgbe {

KeyFrame::~KeyFrame() {}

KeyFrame::KeyFrame(const comm_msgs::keyframeConstPtr &keyframe_msg,
                   const SystemParameters &params, const uint64_t agent_id)
    : parameters_(params),
      voc_ptr_(NULL),
      bow_vec_(DBoW2::BowVector()),
      feat_vec_(DBoW2::FeatureVector()),
      has_covariance_(false),
      has_point_cloud_(false) {
  // Extract the ids.
  uint64_t frame_id = keyframe_msg->frameId;
  // uint64_t agent_id = keyframe_msg->agentId;
  double timestamp = keyframe_msg->header.stamp.toSec();

  // Extract the debug image
  //  image_ = cv_bridge::toCvCopy(keyframe_msg->debugImage,
  //      sensor_msgs::image_encodings::MONO8)->image;

  // Extract the keypoint information
  cv::Mat descriptors = cv_bridge::toCvCopy(keyframe_msg->keyPtsDescriptors,
                                            sensor_msgs::image_encodings::MONO8)
                            ->image;
  const size_t num_kpts = keyframe_msg->numKeyPts;
  Vector2Vector keypoints;
  keypoints.reserve(num_kpts);
  for (size_t i = 0; i < num_kpts; ++i) {
    keypoints.push_back(
        Eigen::Vector2d(keyframe_msg->keyPts[i].x, keyframe_msg->keyPts[i].y));
  }

  // Extract the landmark information
  const size_t num_lms = keyframe_msg->landmarks.size();
  Vector3Vector landmarks;
  landmarks.reserve(num_lms);
  std::vector<size_t> landmark_index;
  landmark_index.reserve(num_lms);
  for (size_t i = 0; i < num_lms; ++i) {
    landmarks.push_back(Eigen::Vector3d(keyframe_msg->landmarks[i].x,
                                        keyframe_msg->landmarks[i].y,
                                        keyframe_msg->landmarks[i].z));
    landmark_index.push_back(keyframe_msg->landmarks[i].index);
  }

  // Extract the connected ids
  std::vector<uint64_t> connections;
  for (size_t i = 0; i < keyframe_msg->connections.size(); ++i) {
    connections.push_back(keyframe_msg->connections[i]);
  }

  // Extract the local odometry
  Eigen::Matrix4d T_O_S = Eigen::Matrix4d::Identity();
  T_O_S.block<3, 3>(0, 0) =
      Eigen::Quaterniond(keyframe_msg->odometry.pose.pose.orientation.w,
                         keyframe_msg->odometry.pose.pose.orientation.x,
                         keyframe_msg->odometry.pose.pose.orientation.y,
                         keyframe_msg->odometry.pose.pose.orientation.z)
          .toRotationMatrix();
  T_O_S(0, 3) = keyframe_msg->odometry.pose.pose.position.x;
  T_O_S(1, 3) = keyframe_msg->odometry.pose.pose.position.y;
  T_O_S(2, 3) = keyframe_msg->odometry.pose.pose.position.z;

  constructKeyFrame(frame_id, agent_id, timestamp, keypoints, descriptors,
                    landmarks, landmark_index, connections, T_O_S);
}

void KeyFrame::constructKeyFrame(
    const uint64_t frame_id, const uint64_t agent_id, const double timestamp,
    const Vector2Vector &keypoints, const cv::Mat &descriptors,
    const Vector3Vector &landmarks, const std::vector<size_t> &kpt_idxs,
    const std::vector<uint64_t> &connections, const Eigen::Matrix4d &T_O_S) {
  // Copy what is straight forward
  id_ = std::make_pair(agent_id, frame_id);
  keypoints_ = keypoints;
  descriptors_ = descriptors.clone();
  landmarks_ = landmarks;
  T_O_S_ = T_O_S;
  timestamp_ = timestamp;
  // Store the camera
  camera_ = parameters_.camera_parameters[agent_id].camera->getCameraShared(0);

  // Extract the vocabulary
  voc_ptr_ = parameters_.voc_ptr;

  // Fill in the index to access lanmdarks by keypoint index.
  landmark_index_.resize(keypoints_.size(), -1);
  for (size_t i = 0; i < kpt_idxs.size(); ++i) {
    landmark_index_[kpt_idxs[i]] = i;
  }

  // Fill in the odometry connections
  for (size_t i = 0; i < connections.size(); ++i) {
    connections_odom_.insert(connections[i]);
  }

  // Assign the keypoints to the grid
  assignFeaturesToGrid();

  // Compute the BoW representation
  computeBoW();
}

aslam::ProjectionResult KeyFrame::projectPoint(const Eigen::Vector3d &l_C,
                                               Eigen::Vector2d &proj) {
  return camera_->project3(l_C, &proj);
}

bool KeyFrame::insertOdomConnection(const uint64_t frame_id) {
  std::unique_lock<std::mutex> lock(mutex_connections_);
  if (connections_odom_.count(frame_id)) {
    return false;
  }

  connections_odom_.insert(frame_id);
  return true;
}

bool KeyFrame::removeOdomConnection(const uint64_t frame_id) {
  std::unique_lock<std::mutex> lock(mutex_connections_);
  if (!connections_odom_.count(frame_id)) {
    return false;
  }
  auto itr =
      std::find(connections_odom_.begin(), connections_odom_.end(), frame_id);
  connections_odom_.erase(itr);
  return true;
}

bool KeyFrame::insertLoopClosureConnection(const Identifier &loop_id) {
  std::unique_lock<std::mutex> lock(mutex_connections_);
  // Sanity check --> if the loop is this keyframe itself
  if (loop_id == id_) {
    return false;
  }

  if (connections_loop_.count(loop_id)) {
    return false;
  }

  connections_loop_.insert(loop_id);
  return true;
}

bool KeyFrame::removeLoopClosureConnection(const Identifier &loop_id) {
  std::unique_lock<std::mutex> lock(mutex_connections_);

  if (!connections_loop_.count(loop_id)) {
    return false;
  }

  auto del_itr =
      std::find(connections_loop_.begin(), connections_loop_.end(), loop_id);
  if (del_itr == connections_loop_.end()) {
    return false;
  }

  connections_loop_.erase(del_itr);
}

std::set<Identifier> KeyFrame::getOdomConnections() {
  std::unique_lock<std::mutex> lock(mutex_connections_);
  std::set<Identifier> connections;
  for (auto itr = connections_odom_.begin(); itr != connections_odom_.end();
       ++itr) {
    connections.insert(std::make_pair(id_.first, (*itr)));
  }
  return connections;
}

std::set<Identifier> KeyFrame::getLoopConnections() {
  std::unique_lock<std::mutex> lock(mutex_connections_);

  return connections_loop_;
}

bool KeyFrame::insertLoopClosureEdge(const LoopEdge &loop_edge) {
  std::unique_lock<std::mutex> lock(mutex_connections_);
  // Check if the connection really corresponds to this frame
  if ((loop_edge.id_A != id_) && (loop_edge.id_B != id_)) {
    return false;
  }

  // Check if the same loop was allready added
  bool could_insert = true;
  for (size_t i = 0; i < loop_edges_.size(); ++i) {
    LoopEdge existing_edge = loop_edges_[i];
    if (((existing_edge.id_A == loop_edge.id_A) &&
         (existing_edge.id_B == loop_edge.id_B)) ||
        ((existing_edge.id_B == loop_edge.id_A) &&
         (existing_edge.id_A == loop_edge.id_B))) {
      could_insert = false;
      break;
    }
  }

  if (could_insert) {
    loop_edges_.push_back(loop_edge);
  }

  return could_insert;
}

bool KeyFrame::removeLoopClosureEdge(const LoopEdge &loop_edge) {
  std::unique_lock<std::mutex> lock(mutex_connections_);

  // Loop over all edges and check if the requested is there.
  bool success = false;
  LoopEdges::iterator del_itr;
  for (auto itr = loop_edges_.begin(); itr != loop_edges_.end(); ++itr) {
    LoopEdge existing_edge = (*itr);
    if (((existing_edge.id_A == loop_edge.id_A) &&
         (existing_edge.id_B == loop_edge.id_B)) ||
        ((existing_edge.id_B == loop_edge.id_A) &&
         (existing_edge.id_A == loop_edge.id_B))) {
      success = true;
      del_itr = itr;
      break;
    }
  }

  if (success) {
    loop_edges_.erase(del_itr);
  }

  return success;
}

Eigen::Vector2d KeyFrame::getKeypoint(const size_t kp_idx) {
  return keypoints_[kp_idx];
}

Eigen::Vector3d KeyFrame::getKeypointBearing(const size_t kp_idx) {
  Eigen::Vector3d bearing;
  camera_->backProject3(keypoints_[kp_idx], &bearing);
  bearing.normalize();
  return bearing;
}

bool KeyFrame::getLandmark(const size_t kp_idx, Eigen::Vector3d &landmark) {
  if (landmark_index_[kp_idx] >= 0) {
    landmark = landmarks_[landmark_index_[kp_idx]];
    return true;
  }

  return false;
}

double KeyFrame::getFocalLength() {
  Eigen::VectorXd intrinsics = camera_->getParameters().transpose();
  double focal_length = (intrinsics[0] + intrinsics[1]) / 2.0;
  return focal_length;
}

Eigen::Matrix4d KeyFrame::getExtrinsics() {
  return (parameters_.camera_parameters[this->id_.first]
              .camera->get_T_C_B(0)
              .getTransformationMatrix())
      .inverse();
}

void KeyFrame::setLoopClosurePose(const Eigen::Matrix4d &T_M_S) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  T_M_S_loop_tmp_ = T_M_S;
}

void KeyFrame::setOptimizedPose(const Eigen::Matrix4d &T_M_S) {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  T_M_S_ = T_M_S;
}

Eigen::Matrix4d KeyFrame::getOptimizedPose() {
  std::unique_lock<std::mutex> lock(mutex_pose_);
  return T_M_S_;
}

void KeyFrame::setOptimizedPoseCovariance(const Eigen::Matrix4d &cov_T_M_S) {
  std::lock_guard<std::mutex> lock(mutex_pose_);
  cov_T_M_S_ = cov_T_M_S;
  has_covariance_ = true;
}

bool KeyFrame::getOptimizedPoseCovariance(Eigen::Matrix4d &cov_T_M_S) {
  std::lock_guard<std::mutex> lock(mutex_pose_);
  cov_T_M_S = cov_T_M_S_;
  return has_covariance_;
}

void KeyFrame::addPclMessage(const sensor_msgs::PointCloud2 &pcl_msg) {
  pointcloud_ = pcl_msg;
  has_point_cloud_ = true;
}

void KeyFrame::addFusedPcl(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud) {
  fused_pcl_cloud_ = *fused_pcl_cloud;
  fused_pcl_cloud_ptr = fused_pcl_cloud;
  has_point_cloud_ = true;
}

bool KeyFrame::getPclMessage(sensor_msgs::PointCloud2 &pcl_msg) {
  if (!has_point_cloud_) return false;

  pcl_msg = pointcloud_;
  return true;
}

void KeyFrame::computeBoW() {
  if (bow_vec_.empty() || feat_vec_.empty()) {
    std::vector<cv::Mat> current_descr = toDescriptorVector();
    // Feature vector associate features with nodes in the 4th level (from
    // leaves up) We assume the vocabulary tree has 6 levels, change the 4
    // otherwise
    voc_ptr_->transform(current_descr, bow_vec_, feat_vec_, 4);
  }
}

void KeyFrame::assignFeaturesToGrid() {
  grid_element_inv_width_ = static_cast<double>(FRAMEGRIDCOLS) /
                            static_cast<double>(camera_->imageWidth());
  grid_element_inv_height_ = static_cast<double>(FRAMEGRIDROWS) /
                             static_cast<double>(camera_->imageHeight());
  int num_reserve = 0.5 * keypoints_.size() / (FRAMEGRIDCOLS * FRAMEGRIDROWS);
  for (size_t i = 0; i < FRAMEGRIDCOLS; ++i)
    for (size_t j = 0; j < FRAMEGRIDROWS; ++j) grid_[i][j].reserve(num_reserve);

  for (int i = 0; i < keypoints_.size(); ++i) {
    int grid_pos_x, grid_pos_y;
    positionInGrid(keypoints_[i], grid_pos_x, grid_pos_y);
    grid_[grid_pos_x][grid_pos_y].push_back(i);
  }
}

void KeyFrame::positionInGrid(const Eigen::Vector2d &kp, int &pos_x,
                              int &pos_y) {
  pos_x = std::floor(kp(0) * grid_element_inv_width_);
  pos_y = std::floor(kp(1) * grid_element_inv_height_);
}

std::vector<cv::Mat> KeyFrame::toDescriptorVector() {
  std::vector<cv::Mat> descriptor_vector;
  descriptor_vector.reserve(descriptors_.rows);
  for (size_t i = 0; i < descriptors_.rows; ++i) {
    descriptor_vector.push_back(descriptors_.row(i));
  }

  return descriptor_vector;
}

void KeyFrame::writeLoopClosureTransform(const std::string &filename,
                                         std::shared_ptr<KeyFrame> lc,
                                         Eigen::Matrix4d &T_A_B) {
  std::lock_guard<std::mutex> lock(mutex_fn);
  std::ofstream file;

  file.open(filename);
  const Eigen::Quaterniond q_A_Bi(T_A_B.block<3, 3>(0, 0));
  const Eigen::Vector3d p_A_Bi = T_A_B.block<3, 1>(0, 3);

  const Eigen::Quaterniond q_M_S(T_M_S_.block<3, 3>(0, 0));
  const Eigen::Vector3d p_M_S = T_M_S_.block<3, 1>(0, 3);
  file << "Timestamp A, agent_id A, frame A, Timestamp B, agent_id B, frame B, "
          "p_A_B_x, p_A_B_y, p_A_B_z,";
  file << "q_A_B_w, q_A_B_x, q_A_B_y, q_A_B_z" << '\n';
  file << std::setprecision(25);
  file << timestamp_ << "," << id_.first << "," << id_.second << ",";
  file << lc->getTimestamp() << "," << lc->getId().first << ","
       << lc->getId().second << ",";
  file << p_A_Bi(0) << "," << p_A_Bi(1) << "," << p_A_Bi(2) << ",";
  file << q_A_Bi.w() << "," << q_A_Bi.x() << "," << q_A_Bi.y() << ",";
  file << q_A_Bi.z() << '\n';
  file.close();
}

}  // namespace pgbe
