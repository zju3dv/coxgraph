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
 * subscriber.cpp
 * @brief Implementation file for the Subscriber Class
 * @author: Marco Karrer
 * Created on: Aug 13, 2018
 */

#include "subscriber.hpp"
#include <random>

namespace pgbe {

Subscriber::Subscriber() {}

Subscriber::~Subscriber() {}

Subscriber::Subscriber(ros::NodeHandle &nh, const SystemParameters &parameters,
                       std::shared_ptr<System> system)
    : nh_(&nh), parameters_(parameters), system_(system) {
  setNodeHandle(nh, parameters);
}

void Subscriber::setNodeHandle(ros::NodeHandle &nh,
                               const SystemParameters &parameters) {
  nh_ = &nh;
  parameters_ = parameters;

  // Setup the callbacks
  sub_keyframe_.reserve(parameters_.num_agents);
  sub_keyframe_.reserve(parameters_.num_agents);
  last_fake_gps_.resize(parameters_.num_agents);
  last_fake_gps2_.resize(parameters_.num_agents);

  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    ros::Subscriber tmp_sub_kf = nh_->subscribe<comm_msgs::keyframe>(
        "/keyframe" + std::to_string(i), 1000,
        boost::bind(&Subscriber::keyFrameCallback, this, _1, i));
    sub_keyframe_.push_back(tmp_sub_kf);

    ros::Subscriber tmp_sub_odom = nh_->subscribe<nav_msgs::Odometry>(
        "/odometry" + std::to_string(i), 1000,
        boost::bind(&Subscriber::odometryCallback, this, _1, i));
    sub_odometry_.push_back(tmp_sub_odom);

    ros::Subscriber tmp_sub_pcl = nh_->subscribe<sensor_msgs::PointCloud2>(
        "/pointcloud" + std::to_string(i), 1000,
        boost::bind(&Subscriber::pointCloudCallback, this, _1, i));
    sub_pointcloud_.push_back(tmp_sub_pcl);

    ros::Subscriber tmp_sub_gps = nh_->subscribe<sensor_msgs::NavSatFix>(
        "/gps" + std::to_string(i), 1000,
        boost::bind(&Subscriber::gpsCallback, this, _1, i));
    sub_gps_.push_back(tmp_sub_gps);

    ros::Subscriber tmp_sub_fake_gps =
        nh_->subscribe<geometry_msgs::TransformStamped>(
            "/fake_gps" + std::to_string(i), 1000,
            boost::bind(&Subscriber::fakeGpsCallback, this, _1, i));
    sub_fake_gps_.push_back(tmp_sub_fake_gps);

    ros::Subscriber tmp_sub_fake_gps2 =
        nh_->subscribe<geometry_msgs::PointStamped>(
            "/fake_gps2_" + std::to_string(i), 1000,
            boost::bind(&Subscriber::fakeGpsCallback2, this, _1, i));
    sub_fake_gps2_.push_back(tmp_sub_fake_gps2);

    ros::Subscriber tmp_sub_fused_pcl = nh_->subscribe<comm_msgs::fused_pcl>(
        "/fused_pcl" + std::to_string(i), 1000,
        boost::bind(&Subscriber::fusedPointCloudCallback, this, _1, i));
    sub_fused_pcl_.push_back(tmp_sub_fused_pcl);
  }
  ROS_INFO("[PGB] Subscribers set");
}

void Subscriber::keyFrameCallback(const comm_msgs::keyframeConstPtr &kf_msg,
                                  const uint64_t agent_id) {
  // Just push the measurement to the system
  system_->addKeyFrameMsg(kf_msg, agent_id);
}

void Subscriber::odometryCallback(const nav_msgs::OdometryConstPtr &odom_msg,
                                  const uint64_t agent_id) {
  // Just push the measurement to the system
  system_->addOdometryMsg(odom_msg, agent_id);
}

void Subscriber::pointCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &pcl_msg, const uint64_t agent_id) {
  // Just push the measurement to the system
  system_->addPointCloudMsg(pcl_msg, agent_id);
}

void Subscriber::fusedPointCloudCallback(
    const comm_msgs::fused_pclConstPtr &fused_pcl_msg,
    const uint64_t agent_id) {
  // Just push the measurement to the system
  system_->addFusedPointCloudMsg(fused_pcl_msg, agent_id);
}

void Subscriber::gpsCallback(const sensor_msgs::NavSatFixConstPtr &gps_msg,
                             const uint64_t agent_id) {
  // Just push the measurement to the system
  system_->addGpsMsg(gps_msg, agent_id);
}

void Subscriber::fakeGpsCallback(
    const geometry_msgs::TransformStampedConstPtr &msg,
    const uint64_t agent_id) {
  if (std::abs(last_fake_gps_[agent_id] - msg->header.stamp.toSec()) < 0.32) {
    return;
  }

  // Create a converter
  geodetic_converter::GeodeticConverter converter;
  converter.initialiseReference(
      parameters_.gps_parameters[agent_id].local_reference[0],
      parameters_.gps_parameters[agent_id].local_reference[1],
      parameters_.gps_parameters[agent_id].local_reference[2]);

  double x = msg->transform.translation.x;
  double y = msg->transform.translation.y;
  double z = msg->transform.translation.z;
  double latitude, longitude, altitude;
  converter.enu2Geodetic(x, y, z, &latitude, &longitude, &altitude);

  // Create a nav msg from it
  sensor_msgs::NavSatFixPtr gps_msg(new sensor_msgs::NavSatFix());
  gps_msg->header.stamp = msg->header.stamp;
  gps_msg->altitude = altitude;
  gps_msg->latitude = latitude;
  gps_msg->longitude = longitude;

  gps_msg->position_covariance[0] = 0.005;
  gps_msg->position_covariance[1] = 0.0;
  gps_msg->position_covariance[2] = 0.0;
  gps_msg->position_covariance[3] = 0.0;
  gps_msg->position_covariance[4] = 0.005;
  gps_msg->position_covariance[5] = 0.0;
  gps_msg->position_covariance[6] = 0.0;
  gps_msg->position_covariance[7] = 0.0;
  gps_msg->position_covariance[8] = 0.005;

  // Just push the measurement to the system
  system_->addGpsMsg(gps_msg, agent_id);
  last_fake_gps_[agent_id] = msg->header.stamp.toSec();
}

void Subscriber::fakeGpsCallback2(
    const geometry_msgs::PointStampedConstPtr &msg, const uint64_t agent_id) {
  if (std::abs(last_fake_gps2_[agent_id] - msg->header.stamp.toSec()) < 0.99) {
    return;
  }

  // Create a converter
  geodetic_converter::GeodeticConverter converter;
  converter.initialiseReference(
      parameters_.gps_parameters[agent_id].local_reference[0],
      parameters_.gps_parameters[agent_id].local_reference[1],
      parameters_.gps_parameters[agent_id].local_reference[2]);

  // Add Gaussian noise to fake GPS measurement (used for EuRoC testing)
  auto dist = std::bind(std::normal_distribution<double>{0.0, 0.1},
                        std::mt19937(std::random_device{}()));

  double x_rand = dist();
  double y_rand = dist();
  double z_rand = dist();
  // double x_rand = 0;
  // double y_rand = 0;
  // double z_rand = 0;

  double x = msg->point.x + x_rand;
  double y = msg->point.y + y_rand;
  double z = msg->point.z + z_rand;
  double latitude, longitude, altitude;
  converter.enu2Geodetic(x, y, z, &latitude, &longitude, &altitude);

  // Create a nav msg from it
  sensor_msgs::NavSatFixPtr gps_msg(new sensor_msgs::NavSatFix());
  gps_msg->header.stamp = msg->header.stamp;
  gps_msg->altitude = altitude;
  gps_msg->latitude = latitude;
  gps_msg->longitude = longitude;

  gps_msg->position_covariance[0] = 0.1;
  gps_msg->position_covariance[1] = 0.0;
  gps_msg->position_covariance[2] = 0.0;
  gps_msg->position_covariance[3] = 0.0;
  gps_msg->position_covariance[4] = 0.1;
  gps_msg->position_covariance[5] = 0.0;
  gps_msg->position_covariance[6] = 0.0;
  gps_msg->position_covariance[7] = 0.0;
  gps_msg->position_covariance[8] = 0.1;

  // Just push the measurement to the system
  system_->addGpsMsg(gps_msg, agent_id);
  last_fake_gps2_[agent_id] = msg->header.stamp.toSec();
}

}  // namespace pgbe
