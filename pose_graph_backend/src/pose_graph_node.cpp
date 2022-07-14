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
 * pose_graph_node.cpp
 * @brief Main executable for the pose_graph_backend.
 * @author: Marco Karrer
 * Created on: Aug 13, 2018
 */

#include <glog/logging.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <memory>

#include "parameter-reader.hpp"
#include "pose_graph_backend/system.hpp"
#include "publisher.hpp"
#include "subscriber.hpp"

int main(int argc, char **argv) {
  // Start logging
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  // Initialize ros
  ros::init(argc, argv, "pose_graph_backend");

  // set up the node
  ros::NodeHandle nh("pose_graph_backend");
  ros::NodeHandle nh_private("~");

  // Read the parameters
  int num_agents = 0;
  if (!nh.getParam("num_agents", num_agents)) {
    ROS_ERROR("[PGB] Cannot get number of agents, abort...");
    return -1;
  }
  ROS_INFO("[PGB] Num agents: %d", num_agents);

  pgbe::ParameterReader param_reader(nh, num_agents);
  pgbe::SystemParameters parameters;
  bool read_params = param_reader.getParameters(parameters);
  if (!read_params) {
    ROS_ERROR("[PGB] Could not read the required parameters, abort...");
    return -1;
  }
  ROS_INFO("[PGB] Sucessfully read the parameters and the vocabulary");

  // Create the system
  std::shared_ptr<pgbe::System> system(
      new pgbe::System(parameters, nh, nh_private));

  // Create the subscriber& publisher
  pgbe::Subscriber subscriber(nh, parameters, system);
  pgbe::Publisher publisher(nh, parameters);

  // Register the publisher callbacks
  system->setTransformCallback(std::bind(
      &pgbe::Publisher::publishTransformsAsCallback, &publisher,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5));

  system->setPCLCallback(std::bind(
      &pgbe::Publisher::publishPCLCallback, &publisher, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3));

  system->setPathCallback(std::bind(&pgbe::Publisher::publishPathCallback,
                                    &publisher, std::placeholders::_1,
                                    std::placeholders::_2));

  system->setCamVizCallback(std::bind(
      &pgbe::Publisher::publishCamVizCallback, &publisher,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  system->setFusedPCLCallback(
      std::bind(&pgbe::Publisher::publishFusedPCLCallback, &publisher,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));

  ROS_INFO("[PGB] Set callbacks");
  ros::spin();
  return 0;
}
