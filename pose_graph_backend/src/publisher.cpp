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
 * publisher.cpp
 * @brief Implementation file for the Publisher Class
 * @author: Marco Karrer
 * Created on: Aug 29, 2018
 */

#include "publisher.hpp"

namespace pgbe {

Publisher::Publisher() {}

Publisher::~Publisher() { tf_thread_.join(); }

Publisher::Publisher(ros::NodeHandle &nh, const SystemParameters &parameters)
    : nh_(&nh), parameters_(parameters) {
  setNodeHandle(nh, parameters);
}

void Publisher::setNodeHandle(ros::NodeHandle &nh,
                              const SystemParameters &parameters) {
  nh_ = &nh;
  parameters_ = parameters;

  // Create the publisher
  transform_pub1_.reserve(parameters_.num_agents);
  transform_pub2_.reserve(parameters_.num_agents);
  transform_pub3_.reserve(parameters_.num_agents);
  pgbe_pointclouds_pub_.reserve(parameters_.num_agents);
  camera_pose_visual_pub_.reserve(parameters_.num_agents);
  camera_pose_visual_.reserve(parameters_.num_agents);
  path_pub_.reserve(parameters_.num_agents);
  fused_pcl_pub_.reserve(parameters_.num_agents);
  pcl_transform_pub_.reserve(parameters_.num_agents);
  msgs_T_M_O_.resize(parameters_.num_agents);
  msgs_T_W_M_.resize(parameters_.num_agents);
  has_transforms_.resize(parameters_.num_agents, false);

  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    ros::Publisher tmp_pub1 = nh_->advertise<geometry_msgs::TransformStamped>(
        "odom_to_map" + std::to_string(i), 1);
    transform_pub1_.push_back(tmp_pub1);
    ros::Publisher tmp_pub2 = nh_->advertise<geometry_msgs::TransformStamped>(
        "map_to_world" + std::to_string(i), 1);
    transform_pub2_.push_back(tmp_pub2);
    ros::Publisher tmp_pub3 = nh_->advertise<geometry_msgs::TransformStamped>(
        "transform" + std::to_string(i), 1);
    transform_pub3_.push_back(tmp_pub3);

    ros::Publisher tmp_pub_pgbe_ptcloud =
        nh_->advertise<sensor_msgs::PointCloud2>("pgbe_pointcloud", 1);
    pgbe_pointclouds_pub_.push_back(tmp_pub_pgbe_ptcloud);

    ros::Publisher tmp_path_pub =
        nh_->advertise<nav_msgs::Path>("path" + std::to_string(i), 1);
    path_pub_.push_back(tmp_path_pub);

    CameraPoseVisualization tmp_camera_pose_visual(1, 0, 0, 1);
    camera_pose_visual_.push_back(tmp_camera_pose_visual);

    ros::Publisher tmp_camera_pose_visual_pub =
        nh_->advertise<visualization_msgs::MarkerArray>(
            "camera_pose_visual" + std::to_string(i), 1);
    camera_pose_visual_pub_.push_back(tmp_camera_pose_visual_pub);

    ros::Publisher tmp_fused_pcl_pb = nh_->advertise<sensor_msgs::PointCloud2>(
        "fused_pcl_world" + std::to_string(i), 1);
    fused_pcl_pub_.push_back(tmp_fused_pcl_pb);

    ros::Publisher tmp_pcl_transform_pub =
        nh_->advertise<comm_msgs::pcl_transform>(
            "pcl_transform" + std::to_string(i), 1);
    pcl_transform_pub_.push_back(tmp_pcl_transform_pub);
  }

  // Launch the TF publisher thread
  tf_thread_ = std::thread(&Publisher::publishTfTransforms, this);
}

void Publisher::publishTransformsAsCallback(const double &timestamp,
                                            const uint64_t agent_id,
                                            const Eigen::Matrix4d T_M_O,
                                            const Eigen::Matrix4d T_W_M,
                                            const Eigen::Matrix4d T_M_C) {
  // Create the pose message odometry-to-world
  geometry_msgs::TransformStamped msg_T_M_O;
  msg_T_M_O.header.stamp = time_obj_.fromSec(timestamp);
  msg_T_M_O.header.frame_id = "map_" + std::to_string(agent_id);
  msg_T_M_O.child_frame_id = "odom_" + std::to_string(agent_id);
  const Eigen::Quaterniond q_M_O(T_M_O.block<3, 3>(0, 0));
  msg_T_M_O.transform.rotation.w = q_M_O.w();
  msg_T_M_O.transform.rotation.x = q_M_O.x();
  msg_T_M_O.transform.rotation.y = q_M_O.y();
  msg_T_M_O.transform.rotation.z = q_M_O.z();
  msg_T_M_O.transform.translation.x = T_M_O(0, 3);
  msg_T_M_O.transform.translation.y = T_M_O(1, 3);
  msg_T_M_O.transform.translation.z = T_M_O(2, 3);

  // Create the pose message world-to-reference
  geometry_msgs::TransformStamped msg_T_W_M;
  msg_T_W_M.header.stamp = time_obj_.fromSec(timestamp);
  msg_T_W_M.header.frame_id = "world";
  msg_T_W_M.child_frame_id = "map_" + std::to_string(agent_id);
  const Eigen::Quaterniond q_W_M(T_W_M.block<3, 3>(0, 0));
  msg_T_W_M.transform.rotation.w = q_W_M.w();
  msg_T_W_M.transform.rotation.x = q_W_M.x();
  msg_T_W_M.transform.rotation.y = q_W_M.y();
  msg_T_W_M.transform.rotation.z = q_W_M.z();
  msg_T_W_M.transform.translation.x = T_W_M(0, 3);
  msg_T_W_M.transform.translation.y = T_W_M(1, 3);
  msg_T_W_M.transform.translation.z = T_W_M(2, 3);

  // Create the pose message odometry-to-world
  geometry_msgs::TransformStamped msg_T_W_O;
  msg_T_W_O.header.stamp = time_obj_.fromSec(timestamp);
  msg_T_W_O.header.frame_id = "world";
  msg_T_W_O.child_frame_id = "odom_" + std::to_string(agent_id);
  const Eigen::Matrix4d T_W_O = T_W_M * T_M_O;
  const Eigen::Quaterniond q_W_O(T_W_O.block<3, 3>(0, 0));
  msg_T_W_O.transform.rotation.w = q_W_O.w();
  msg_T_W_O.transform.rotation.x = q_W_O.x();
  msg_T_W_O.transform.rotation.y = q_W_O.y();
  msg_T_W_O.transform.rotation.z = q_W_O.z();
  msg_T_W_O.transform.translation.x = T_W_O(0, 3);
  msg_T_W_O.transform.translation.y = T_W_O(1, 3);
  msg_T_W_O.transform.translation.z = T_W_O(2, 3);

  // Publish the transformation
  transform_pub1_[agent_id].publish(msg_T_M_O);
  transform_pub2_[agent_id].publish(msg_T_W_M);
  transform_pub3_[agent_id].publish(msg_T_W_O);

  geometry_msgs::TransformStamped msg_T_M_C;
  msg_T_M_C.header.stamp = time_obj_.fromSec(timestamp);
  msg_T_M_C.header.frame_id = "map_" + std::to_string(agent_id);
  msg_T_M_C.child_frame_id = "cam" + std::to_string(agent_id);
  const Eigen::Quaterniond q_M_C(T_M_C.block<3, 3>(0, 0));
  msg_T_M_C.transform.rotation.w = q_M_C.w();
  msg_T_M_C.transform.rotation.x = q_M_C.x();
  msg_T_M_C.transform.rotation.y = q_M_C.y();
  msg_T_M_C.transform.rotation.z = q_M_C.z();
  msg_T_M_C.transform.translation.x = T_M_C(0, 3);
  msg_T_M_C.transform.translation.y = T_M_C(1, 3);
  msg_T_M_C.transform.translation.z = T_M_C(2, 3);

  // Store that we have a transform
  msgs_T_M_O_[agent_id] = msg_T_M_O;
  msgs_T_W_M_[agent_id] = msg_T_W_M;
  has_transforms_[agent_id] = true;

  // Also broadcast the TFs
  //  tf_pub_.sendTransform(msg_T_W_M);
  //  tf_pub_.sendTransform(msg_T_M_O);
}

void Publisher::publishPCLCallback(const double &timestamp,
                                   const uint64_t agent_id,
                                   const Result result) {
  sensor_msgs::PointCloud2 pcl_msg = result.point_clouds[0];
  pgbe_pointclouds_pub_[agent_id].publish(pcl_msg);
}

void Publisher::publishFusedPCLCallback(
    const uint64_t agent_id, const double &timestamp,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud,
    const Eigen::Matrix4d &T_W_C) {
  //      std::lock_guard<std::mutex> lock(pcl_mutex_);

  geometry_msgs::TransformStamped msg_world;
  msg_world.header.frame_id = "world";
  msg_world.child_frame_id = "pcl_cam" + std::to_string(agent_id);
  // msg_world.header.stamp = time_obj_.fromSec(timestamp);
  msg_world.header.stamp = ros::Time::now();

  Eigen::Matrix4d T_W_C_anchor = T_W_C;
  const Eigen::Quaterniond q_W_C_anchor(T_W_C_anchor.block<3, 3>(0, 0));
  msg_world.transform.rotation.w = q_W_C_anchor.w();
  msg_world.transform.rotation.x = q_W_C_anchor.x();
  msg_world.transform.rotation.y = q_W_C_anchor.y();
  msg_world.transform.rotation.z = q_W_C_anchor.z();
  msg_world.transform.translation.x = T_W_C_anchor(0, 3);
  msg_world.transform.translation.y = T_W_C_anchor(1, 3);
  msg_world.transform.translation.z = T_W_C_anchor(2, 3);

  // publish tf
  tf::TransformBroadcaster tf_pcl;
  tf_pcl.sendTransform(msg_world);

  // publish pointcloud
  sensor_msgs::PointCloud2 fused_pcl_msg;
  pcl::toROSMsg(*fused_pcl_cloud, fused_pcl_msg);
  // fused_pcl_msg.header.stamp = time_obj_.fromSec(timestamp);
  fused_pcl_msg.header.stamp = ros::Time::now();
  fused_pcl_msg.header.frame_id = "pcl_cam" + std::to_string(agent_id);
  fused_pcl_pub_[agent_id].publish(fused_pcl_msg);

  comm_msgs::pcl_transform pcl_transform_msg;
  pcl_transform_msg.header.stamp = ros::Time::now();
  // pcl_transform_msg.header.frame_id = "pcl_cam" + std::to_string(agent_id);
  pcl_transform_msg.fusedPointcloud = fused_pcl_msg;
  pcl_transform_msg.worldTransform = msg_world;

  pcl_transform_pub_[agent_id].publish(pcl_transform_msg);
}

void Publisher::publishPathCallback(nav_msgs::Path &msg_path,
                                    const uint64_t agent_id) {
  path_pub_[agent_id].publish(msg_path);
}

void Publisher::publishCamVizCallback(const uint64_t agent_id,
                                      const double &timestamp,
                                      const Eigen::Matrix4d T_W_C) {
  visualization_msgs::MarkerArray markerArray_msg;
  Eigen::Vector3d p_W_C(T_W_C(0, 3), T_W_C(1, 3), T_W_C(2, 3));
  Eigen::Quaterniond q_W_C(T_W_C.block<3, 3>(0, 0));
  camera_pose_visual_[agent_id].reset();
  camera_pose_visual_[agent_id].add_pose(p_W_C, q_W_C);

  std_msgs::Header header;
  header.stamp = ros::Time(timestamp);
  header.frame_id = "world";
  camera_pose_visual_[agent_id].publish_by(camera_pose_visual_pub_[agent_id],
                                           header);
}

void Publisher::publishTfTransforms() {
  while (ros::ok()) {
    for (size_t i = 0; i < parameters_.num_agents; ++i) {
      if (!has_transforms_[i]) {
        continue;
      }
      geometry_msgs::TransformStamped msg_T_W_Mi = msgs_T_W_M_[i];
      msg_T_W_Mi.header.stamp = ros::Time::now();
      //      msg_T_W_Mi.header.frame_id = "world";
      //      msg_T_W_Mi.child_frame_id = "map_" + std::to_string(i);
      geometry_msgs::TransformStamped msg_T_M_Oi = msgs_T_M_O_[i];
      msg_T_M_Oi.header.stamp = ros::Time::now();
      //      msg_T_M_Oi.header.frame_id = "map_" + std::to_string(i);
      //      msg_T_M_Oi.child_frame_id = "odom_" + std::to_string(i);

      tf_pub_.sendTransform(msg_T_W_Mi);
      tf_pub_.sendTransform(msg_T_M_Oi);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

}  // namespace pgbe
