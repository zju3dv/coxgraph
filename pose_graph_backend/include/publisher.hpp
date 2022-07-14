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
 * publisher.hpp
 * @brief Header file for the Publisher Class
 * @author: Marco Karrer
 * Created on: Aug 29, 2018
 */

#pragma once

#include <deque>
#include <memory>

#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf/transform_broadcaster.h>

#include <comm_msgs/keyframe.h>
#include <comm_msgs/pcl_transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <pcl/common/transforms.h>
#include "CameraPoseVisualization.h"
#include "parameters.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pose_graph_backend/system.hpp"

/// \brief pgbe Main namespace of this package
namespace pgbe {

class Publisher {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  /// \brief Default constructor.
  Publisher();
  ~Publisher();

  /// \brief Constructor with parameters.
  /// @param nh The ros node-handle.
  Publisher(ros::NodeHandle& nh, const SystemParameters& parameters);

  /// \brief Set the node handle. This sets up the callbacks.
  /// @param nh The node handle.
  void setNodeHandle(ros::NodeHandle& nh, const SystemParameters& parameters);

  /// \brief Publish transformations
  /// @param timestamp The timestamp.
  /// @param agent_id The agent for which it should be published.
  /// @param T_M_O The odometry frame transformation.
  /// @param T_W_M The world transformation.
  void publishTransformsAsCallback(const double& timestamp,
                                   const uint64_t agent_id,
                                   const Eigen::Matrix4d T_M_O,
                                   const Eigen::Matrix4d T_W_M,
                                   const Eigen::Matrix4d T_M_Si);

  // / \brief Call service to planner in case of loop closure
  // / @param timestamp The timestamp.
  // / @param agent_id The agent for which it should be published.
  // /// @param result The result from the optimization.
  void publishFullCallback(const double& timestamp, const uint64_t agent_id,
                           const Result result);

  void publishPCLCallback(const double& timestamp, const uint64_t agent_id,
                          const Result result);

  /// \brief Publish Fused Pointcloud
  /// @param timestamp The timestamp.
  /// @param agent_id The agent for which it should be published.
  /// @param fused_pcl_cloud The cloud to publish.
  /// @param T_W_C The world to camera transformation.
  void publishFusedPCLCallback(
      const uint64_t agent_id, const double& timestamp,
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud,
      const Eigen::Matrix4d& T_W_C);

  /// \brief Publish Path for visualization
  /// @param msg_path The path msg.
  /// @param agent_id The agent for which it should be published.
  void publishPathCallback(nav_msgs::Path& msg_path, const uint64_t agent_id);

  /// \brief Publish camera visualizationn
  /// @param timestamp Timestamp
  /// @param agent_id The agent for which it should be published.
  /// @param agent_id The agent for which it should be published.
  void publishCamVizCallback(const uint64_t agent_id, const double& timestamp,
                             const Eigen::Matrix4d T_W_C);

  /// \brief Publish the TF's at a constant rate
  void publishTfTransforms();

 protected:
  ros::NodeHandle* nh_;

  ros::Time time_obj_;
  std::vector<ros::Publisher> transform_pub1_;
  std::vector<ros::Publisher> transform_pub2_;
  std::vector<ros::Publisher> transform_pub3_;
  std::vector<ros::Publisher> pgbe_pointclouds_pub_;
  std::vector<ros::Publisher> path_pub_;
  std::vector<ros::Publisher> camera_pose_visual_pub_;
  std::vector<ros::Publisher> fused_pcl_pub_;
  std::vector<ros::Publisher> pcl_transform_pub_;
  tf::TransformBroadcaster tf_pub_;

  // Store the tf's for constant rate publishing
  std::thread tf_thread_;
  std::vector<bool> has_transforms_;
  std::vector<geometry_msgs::TransformStamped> msgs_T_M_O_;
  std::vector<geometry_msgs::TransformStamped> msgs_T_W_M_;

  std::mutex pcl_mutex_;

  SystemParameters parameters_;

  std::vector<CameraPoseVisualization> camera_pose_visual_;
};

}  // namespace pgbe
