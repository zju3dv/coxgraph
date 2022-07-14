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
 * subscriber.hpp
 * @brief Header file for the Subscriber Class
 * @author: Marco Karrer
 * Created on: Aug 13, 2018
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

#include <comm_msgs/keyframe.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>

#include "parameters.hpp"
#include "pose_graph_backend/system.hpp"

/// \brief pgbe Main namespace of this package
namespace pgbe {

class Subscriber {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  /// \brief Default constructor.
  Subscriber();
  ~Subscriber();

  /// \brief Constructor with parameters.
  /// @param nh The ros node-handle.
  /// @param parameters The system parameters.
  Subscriber(ros::NodeHandle& nh, const SystemParameters& parameters,
             std::shared_ptr<System> system);

  /// \brief Set the node handle. This sets up the callbacks.
  /// @param nh The node handle.
  void setNodeHandle(ros::NodeHandle& nh, const SystemParameters& parameters);

 protected:
  /// \brief Keyframe callback.
  /// @param kf_msg The keyframe message.
  /// @param pointcloud_msg The pointcloud associated to the keyframe.
  void keyFrameCallback(const comm_msgs::keyframeConstPtr& kf_msg,
                        const uint64_t agent_id);

  /// \brief Odometry Callback.
  /// @param odom_msg The odometry message.
  /// @param agent_id The agent from which the odometry is.
  void odometryCallback(const nav_msgs::OdometryConstPtr& odom_msg,
                        const uint64_t agent_id);

  /// \brief Pointcloud Callback.
  /// @param pcl_msg The point cloud message.
  /// @param agent_id The agent from which the point cloud is.
  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& pcl_msg,
                          const uint64_t agent_id);

  void fusedPointCloudCallback(
      const comm_msgs::fused_pclConstPtr& fused_pcl_msg,
      const uint64_t agent_id);

  /// \brief GPS measurement callback.
  /// @param gps_msg The gps measurement.
  /// @param agent_id The agent from which the gps measurements are.
  void gpsCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg,
                   const uint64_t agent_id);

  /// \brief Fake GPS measurement (alias Vicon).
  /// @param msg The vicon measurement.
  /// @param agent_id The agent id from which the gps measuremens is from
  void fakeGpsCallback(const geometry_msgs::TransformStampedConstPtr& msg,
                       const uint64_t agent_id);

  /// \brief Fake GPS measurement (alias Leica).
  /// @param msg The vicon measurement.
  /// @param agent_id The agent id from which the gps measuremens is from
  void fakeGpsCallback2(const geometry_msgs::PointStampedConstPtr& msg,
                        const uint64_t agent_id);

  ros::NodeHandle* nh_;
  std::vector<ros::Subscriber> sub_keyframe_;
  std::vector<ros::Subscriber> sub_odometry_;
  std::vector<ros::Subscriber> sub_pointcloud_;
  std::vector<ros::Subscriber> sub_fused_pcl_;
  std::vector<ros::Subscriber> sub_gps_;
  std::vector<ros::Subscriber> sub_fake_gps_;
  std::vector<ros::Subscriber> sub_fake_gps2_;

  // Store the last timestamp of the fake_gps
  std::vector<double> last_fake_gps_;
  std::vector<double> last_fake_gps2_;

  SystemParameters parameters_;
  std::shared_ptr<System> system_;
};

}  // namespace pgbe
