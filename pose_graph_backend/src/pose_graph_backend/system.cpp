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
 * system.cpp
 * @brief Source file for the System Class
 * @author: Marco Karrer
 * Created on: Aug 14, 2018
 */

#include "pose_graph_backend/system.hpp"

#include <coxgraph_mod/vio_interface.h>
#include <pcl_conversions/pcl_conversions.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/local-parameterization/pose-quaternion-local-param.h>
#include <robopt_open/local-parameterization/pose-quaternion-yaw-local-param.h>
#include <robopt_open/posegraph-error/four-dof-between.h>
#include <robopt_open/posegraph-error/four-dof-prior-autodiff.h>
#include <robopt_open/posegraph-error/gps-error-autodiff.h>
#include <robopt_open/posegraph-error/six-dof-between.h>
#include <robopt_open/reprojection-error/relative-euclidean.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pose_graph_backend/optimizer.hpp"

namespace pgbe {

System::~System() {
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    keyframe_msgs_received_[i]->Shutdown();
    gps_msgs_received_[i]->Shutdown();
  }

  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    keyframe_consumer_threads_[i].join();
    gps_consumer_threads_[i].join();
  }
}

System::System(const SystemParameters& params, const ros::NodeHandle& nh,
               const ros::NodeHandle& nh_private)
    : parameters_(params),
      nh_private_(nh_private),
      vio_interface_(new coxgraph::mod::VIOInterface(nh, nh_private)) {
  nh_private_.param<std::string>("trajectory_path", file_path_, "");
  LOG(INFO) << "trajectory_path: " << file_path_;
  save_path_srv_ = nh_private_.advertiseService(
      "save_path", &System::savePathCallback, this);
  database_ = std::make_shared<KeyFrameDatabase>(params);
  kf_loop_detection_skip_ = 0;
  init();
}

void System::addKeyFrameMsg(const comm_msgs::keyframeConstPtr& keyframe_msg,
                            const uint64_t agent_id) {
  //  const uint64_t agent_id = keyframe_msg->agentId;
  keyframe_msgs_received_[agent_id]->Push(keyframe_msg);

  // coxgraph::mod::updateNeedToFuse();
}

void System::addOdometryMsg(const nav_msgs::OdometryConstPtr& keyframe_msg,
                            const uint64_t agent_id) {
  // Convert to internal representaion
  const double timestamp = keyframe_msg->header.stamp.toSec();
  Eigen::Vector3d translation(keyframe_msg->pose.pose.position.x,
                              keyframe_msg->pose.pose.position.y,
                              keyframe_msg->pose.pose.position.z);
  Eigen::Quaterniond rotation(keyframe_msg->pose.pose.orientation.w,
                              keyframe_msg->pose.pose.orientation.x,
                              keyframe_msg->pose.pose.orientation.y,
                              keyframe_msg->pose.pose.orientation.z);
  OdomMeasurement meas(timestamp, translation, rotation);
  {
    std::unique_lock<std::mutex> lock(odom_mutex_);
    odom_queues_[agent_id].push_back(meas);
    // std::cout << std::setprecision(14) << "ODM msg: " << timestamp <<
    // std::endl;
  }
}

void System::addPointCloudMsg(const sensor_msgs::PointCloud2ConstPtr& pcl_msg,
                              const uint64_t agent_id) {
  // Add to measurement queue
  pcl_msgs_received_[agent_id]->PushBlockingIfFull(pcl_msg, 1);
}

void System::addFusedPointCloudMsg(
    const comm_msgs::fused_pclConstPtr& fused_pcl_msg,
    const uint64_t agent_id) {
  fused_pcl_msgs_received_[agent_id]->PushNonBlockingDroppingIfFull(
      fused_pcl_msg, 10);
}

void System::addGpsMsg(const sensor_msgs::NavSatFixConstPtr& gps_msg,
                       const uint64_t agent_id) {
  gps_msgs_received_[agent_id]->PushBlockingIfFull(gps_msg, 1);
}

void System::init() {
  // Initialize the maps
  trigger_init_opt_.resize(parameters_.num_agents, false);
  optimization_flags_.resize(parameters_.num_agents, NULL);
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    maps_.emplace_back(std::shared_ptr<Map>(new Map(parameters_, i)));
    maps_[i]->setGPSStatus(
        parameters_.gps_active[i]);  // Embed this into the map initialization
    ROS_INFO_STREAM("[PGB] Agent "
                    << i << " has gps status: " << maps_[i]->getGPSStatus());
    optimization_flags_[i] = new bool;
    *optimization_flags_[i] = false;
  }

  // paths_.resize(parameters_.num_agents,)
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    nav_msgs::Path tmp_path;
    paths_.push_back(tmp_path);
  }
  // Initialize the keyframe database
  database_ = std::make_shared<KeyFrameDatabase>(parameters_);

  // Initialize the loop-detectors
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    loop_detectors_.emplace_back(std::shared_ptr<LoopDetection>(
        new LoopDetection(parameters_, maps_[i], database_, vio_interface_)));
    last_loop_closure_.push_back(0.0);
  }

  // Initialize the GPS conversion nodes
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    gps_converters_.emplace_back(
        std::shared_ptr<geodetic_converter::GeodeticConverter>(
            new geodetic_converter::GeodeticConverter()));
    gps_converters_[i]->initialiseReference(
        parameters_.gps_parameters[i].local_reference[0],
        parameters_.gps_parameters[i].local_reference[1],
        parameters_.gps_parameters[i].local_reference[2]);
  }

  // Initialize the measurement queues
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    keyframe_buffers_.emplace_back(
        std::deque<std::shared_ptr<KeyFrame>,
                   Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>());
    odom_queues_.emplace_back(OdomMeasurementQueue());
    gps_queues_.emplace_back(GPSmeasurementQueue());
    odom_gps_queues_.emplace_back(OdomGPScombinedQueue());
    odom_gps_init_queues_.emplace_back(OdomGPScombinedQueue());
    pcl_queues_.emplace_back(PclMeasurementQueue());
    odom_pcl_queues_.emplace_back(OdomPclCombinedQueue());
    failed_gps_sync_queues_.emplace_back(GPSmeasurementQueue());
    pcl_pub_kf_id_queues_.emplace_back(std::deque<uint64_t>());

    keyframe_msgs_received_.emplace_back(
        KeyFrameMsgQueuePtr(new KeyFrameMsgQueue()));
    odom_msgs_received_.emplace_back(
        OdometryMsgQueuePtr(new OdometryMsgQueue()));
    pcl_msgs_received_.emplace_back(PclMsgQueuePtr(new PclMsgQueue()));
    gps_msgs_received_.emplace_back(GpsMsgQueuePtr(new GpsMsgQueue()));
    keyframes_received_.emplace_back(KeyFrameQueuePtr(new KeyFrameQueue()));
    pub_msgs_received_.emplace_back(ResultQueuePtr(new ResultQueue()));
    fused_pcl_msgs_received_.emplace_back(
        FusedPclMsgQueuePtr(new FusedPclMsgQueue()));
    fused_pcl_msgs_buffer_.emplace_back(FusedPclMsgDequeue());
  }

  // Initialize and start the threads
  for (size_t i = 0; i < parameters_.num_agents; ++i) {
    keyframe_consumer_threads_.emplace_back(&System::keyframeConsumerLoop, this,
                                            i);
    keyframe_optimizer_threads_.emplace_back(&System::optimizerLoop, this, i);
    // gps_consumer_threads_.emplace_back(&System::gpsConsumerLoop, this,
    // i);
    // // No longer using regular pcl setup, everything uses FusedPcl
    // // pcl_consumrer_threads_.emplace_back(
    // //       &System::pclConsumerLoop, this, i);
    // fused_pcl_consumer_threads_.emplace_back(&System::fusedPclConsumerLoop,
    //                                          this, i);
    publisher_threads_.emplace_back(&System::publisherLoop, this, i);
  }

  ROS_INFO("[PGB] Started all threads");
}

void System::keyframeConsumerLoop(const uint64_t agent_id) {
  comm_msgs::keyframeConstPtr keyframe_msg;
  for (;;) {
    if (keyframe_msgs_received_[agent_id]->PopTimeout(&keyframe_msg,
                                                      500000000) == false) {
      continue;
    }

    // Create a new keyframe
    std::shared_ptr<KeyFrame> new_keyframe =
        std::make_shared<KeyFrame>(keyframe_msg, parameters_, agent_id);

    // Add this keyframe to the buffer (in order to have 1 KF delay to assign
    // the incoming other measurements such as GPS), extract the information
    // necessary to process the last Keyframe and update the buffer.
    std::shared_ptr<KeyFrame> keyframe_to_process;
    double time_start, time_end;
    {
      std::lock_guard<std::mutex> lock(kf_mutex_);
      if (keyframe_buffers_[agent_id].empty()) {
        keyframe_buffers_[agent_id].push_back(new_keyframe);
        continue;
      } else if (keyframe_buffers_[agent_id].size() == 1) {
        keyframe_buffers_[agent_id].push_back(new_keyframe);
        keyframe_to_process = keyframe_buffers_[agent_id].front();
        time_start = keyframe_to_process->getTimestamp();
        const double newer_timestamp = new_keyframe->getTimestamp();
        time_end = time_start + (newer_timestamp - time_start) / 2.0;
      } else {
        keyframe_to_process = keyframe_buffers_[agent_id].back();

        // Extract the neighbouring timestamps
        const double older_timestamp =
            keyframe_buffers_[agent_id].front()->getTimestamp();
        const double newer_timestamp = new_keyframe->getTimestamp();
        const double current_timestamp = keyframe_to_process->getTimestamp();
        time_start =
            current_timestamp - (current_timestamp - older_timestamp) / 2.0;
        time_end =
            current_timestamp + (newer_timestamp - current_timestamp) / 2.0;
        // Compute the mid-timestamps for the data association
        keyframe_buffers_[agent_id].push_back(new_keyframe);
        keyframe_buffers_[agent_id].pop_front();
      }
    }

    bool only_insert = false;
    if ((keyframe_to_process->getTimestamp() - last_loop_closure_[agent_id]) <
        6.0) {
      only_insert = true;
    }

    // Don't attempt loop detection if session has just begun (no keyframes)
    Map::KFvec recent_kf = maps_[agent_id]->getMostRecentN(2);
    if (recent_kf.empty()) {
      only_insert = true;
    }

    // Only process loop detections + add to keyframe database for some
    // keyframes
    kf_loop_detection_skip_ =
        (kf_loop_detection_skip_ + 1) % parameters_.loop_detect_skip_kf;

    auto start = chrono::steady_clock::now();
    bool loop_detected;
    if (kf_loop_detection_skip_ == 0) {
      loop_detected = loop_detectors_[agent_id]->addKeyframe(
          keyframe_to_process, only_insert);
    } else {
      loop_detected = false;
    }

    Eigen::Matrix4d T_M_O = maps_[agent_id]->getOdomToMap();
    auto end = chrono::steady_clock::now();
    Identifier kf_id = keyframe_to_process->getId();
 // std::cout
 //     << "Loop_detection for agent " << agent_id << " took: "
 //     << chrono::duration_cast<chrono::milliseconds>(end - start).count()
 //     << " ms"
 //     << std::endl;  // "\t Landmarks: " <<
                       // keyframe_to_process->getNumLandmarks() << std::endl;

    // Update the poses
    Eigen::Matrix4d T_O_Si = keyframe_to_process->getOdometryPose();
    Eigen::Matrix4d T_M_Si = T_M_O * T_O_Si;
    keyframe_to_process->setLoopClosurePose(T_M_Si);
    keyframe_to_process->setOptimizedPose(T_M_Si);
    maps_[agent_id]->addKeyFrame(keyframe_to_process);
  }
}

void System::optimizerLoop(const uint64_t agent_id) {
  std::shared_ptr<KeyFrame> keyframe_ptr;
  for (;;) {
    if (keyframes_received_[agent_id]->PopBlocking(&keyframe_ptr) == false) {
      return;
    }
    // Check if we had a loop closure
    if (!keyframe_ptr->getLoopClosureEdges().empty() ||
        trigger_init_opt_[agent_id]) {
      // Add agents to each other's merged lists
      LoopEdges kf_loop_edges = keyframe_ptr->getLoopClosureEdges();
      for (size_t i = 0; i < kf_loop_edges.size(); ++i) {
        uint64_t other_agent = kf_loop_edges[i].id_B.first;
        if (kf_loop_edges[i].id_A.first != other_agent) {
          maps_[agent_id]->addMergedAgents(other_agent);
          maps_[other_agent]->addMergedAgents(agent_id);

          std::vector<uint64_t> other_agent_merged_list =
              maps_[other_agent]->getMergedAgents();
          for (size_t i = 0; i < other_agent_merged_list.size(); ++i) {
            maps_[agent_id]->addMergedAgents(other_agent_merged_list[i]);
          }
        }
      }

      // Check merged agent lists of other agents
      std::vector<uint64_t> current_merged_maps =
          maps_[agent_id]->getMergedAgents();
      for (auto iter = current_merged_maps.begin();
           iter != current_merged_maps.end(); iter++) {
        std::vector<uint64_t> other_agents_maps =
            maps_[*iter]->getMergedAgents();
        for (auto iter = other_agents_maps.begin();
             iter != other_agents_maps.end(); iter++) {
          maps_[agent_id]->addMergedAgents(*iter);
        }
      }

      {
        std::lock_guard<std::mutex> lock(opt_mutex_);
        Optimizer::MapVec tmp_map_agent_id;

        // Loop through current merged agents to add to optimization
        std::vector<uint64_t> merged_agents =
            maps_[agent_id]->getMergedAgents();
        std::cout << "Optimizing over agents: " << agent_id << ": ";
        for (size_t i = 0; i < merged_agents.size(); ++i) {
          uint64_t agent_map_to_add = merged_agents[i];
          tmp_map_agent_id.push_back(maps_[agent_map_to_add]);
          std::cout << agent_map_to_add << "  ";
        }
        std::cout << std::endl;
        auto start = chrono::steady_clock::now();
        Optimizer::optimizeMapPoseGraph(
            tmp_map_agent_id, optimization_flags_[agent_id], parameters_);
        auto end = chrono::steady_clock::now();

        std::cout
            << "Global Optimization for agent " << agent_id << " took: "
            << chrono::duration_cast<chrono::milliseconds>(end - start).count()
            << " ms" << std::endl;

        trigger_init_opt_[agent_id] = false;
      }

      // Iterate over keyframes
      Eigen::Matrix4d T_W_M = Eigen::Matrix4d::Identity();
      if (maps_[agent_id]->hasValidWorldTransformation()) {
        T_W_M = maps_[agent_id]->getWorldTransformation();
      }

      // Create the result of the optimization
      Map::KFvec recent_kf = maps_[agent_id]->getMostRecentN(2);
      Result result;
      if (!recent_kf.empty()) {
        const double kf_timestamp = recent_kf[0]->getTimestamp();
        result.timestamp = kf_timestamp;
        result.T_W_M = T_W_M;
        result.T_M_O = maps_[agent_id]->getOdomToMap();
        // result(kf_timestamp, maps_[agent_id]->getOdomToMap(), T_W_M);
      }

      // Write transformed trajectory of agents to file for global error
      // comparison
      if (parameters_.logging) {
        for (size_t i = 0; i < parameters_.num_agents; ++i) {
          std::string filename = parameters_.log_folder + "/global_opt_pose_";
          if (maps_[i]->hasValidWorldTransformation()) {
            Map::KFvec last_keyframe = maps_[i]->getMostRecentN(2);
            if (maps_[i]->getMapSize() > 1) {
              std::cout << "Writing global opt poses for agent: " << i
                        << std::endl;
              filename = filename + std::to_string(i) + "_" +
                         std::to_string(last_keyframe[0]->getId().second) +
                         ".csv";
              maps_[i]->writePosesToFileInWorld(filename);
            }
          }
        }
      }
      pub_msgs_received_[agent_id]->PushBlockingIfFull(result, 2);
    }
  }
  updatePath(agent_id);
}

void System::gpsConsumerLoop(const uint64_t agent_id) {
  sensor_msgs::NavSatFixConstPtr gps_msg;
  if (std::all_of(parameters_.gps_active.begin(), parameters_.gps_active.end(),
                  [](bool b) { return b == false; })) {
    if (agent_id == 0) {
      maps_[agent_id]->setWorldAnchor();
      Eigen::Matrix4d T_W_M = Eigen::Matrix4d::Identity();
      Eigen::Matrix4d covariance = Eigen::MatrixXd::Zero(4, 4);
      covariance(0, 0) = covariance(1, 1) = covariance(2, 2) =
          covariance(3, 3) = 0.001;
      maps_[agent_id]->setWorldTransformation(T_W_M, covariance);
    }
  }
  for (;;) {
    if (gps_msgs_received_[agent_id]->PopBlocking(&gps_msg) == false) {
      return;
    }
    // Extract the GPS measurement
    const double timestamp = gps_msg->header.stamp.toSec();
    Eigen::Vector3d raw_measurement, local_measurement;
    raw_measurement[0] = gps_msg->latitude;
    raw_measurement[1] = gps_msg->longitude;
    raw_measurement[2] = gps_msg->altitude;

    Eigen::Matrix3d covariance;
    covariance.setZero();
    if (agent_id != 0) {
      (gps_msg->position_covariance[0] > 0)
          ? covariance(0, 0) = gps_msg->position_covariance[0] * 1.0
          : covariance(0, 0) = 0.1;
      (gps_msg->position_covariance[4] > 0)
          ? covariance(1, 1) = gps_msg->position_covariance[4] * 1.0
          : covariance(1, 1) = 0.1;
      (gps_msg->position_covariance[8] > 0)
          ? covariance(2, 2) = gps_msg->position_covariance[8] * 10.0
          : covariance(2, 2) = 0.1;
    } else {
      (gps_msg->position_covariance[0] > 0)
          ? covariance(0, 0) = gps_msg->position_covariance[0] * 1.0
          : covariance(0, 0) = 0.1;
      (gps_msg->position_covariance[4] > 0)
          ? covariance(1, 1) = gps_msg->position_covariance[4] * 1.0
          : covariance(1, 1) = 0.1;
      (gps_msg->position_covariance[8] > 0)
          ? covariance(2, 2) = gps_msg->position_covariance[8] * 1.0
          : covariance(2, 2) = 0.1;
    }

    // Convert into the local frame
    double local_x, local_y, local_z;
    gps_converters_[agent_id]->geodetic2Enu(
        gps_msg->latitude, gps_msg->longitude, gps_msg->altitude, &local_x,
        &local_y, &local_z);
    if (parameters_.simulation) {
      local_measurement[0] = local_y;
      local_measurement[1] = -local_x;
      local_measurement[2] = local_z;
    } else {
      local_measurement[0] = local_x;
      local_measurement[1] = local_y;
      local_measurement[2] = local_z;
    }
    GPSmeasurement gps_measurement(timestamp, raw_measurement,
                                   local_measurement);
    gps_measurement.covariance = covariance;
    {
      std::lock_guard<std::mutex> lock(gps_mutex_);
      gps_queues_[agent_id].push_back(gps_measurement);
    }

    // If available associate odometry measurements
    OdomMeasurementQueue close_odoms =
        getOdomMeasurements(timestamp - 0.5, timestamp + 0.5, agent_id);
    if (!close_odoms.empty()) {
      syncAndAlignGPS(agent_id, gps_measurement, close_odoms);
      deleteGpsMeasurements(gps_measurement.timestamp, agent_id);
    } else {
      // add gps_measurement to failed queue so it can be tried again
      failed_gps_sync_queues_[agent_id].push_back(gps_measurement);
      while ((gps_measurement.timestamp -
              failed_gps_sync_queues_[agent_id].front().timestamp) > 2.0) {
        failed_gps_sync_queues_[agent_id].pop_front();
      }
    }

    // Attempt to sync previous failed kf
    if (!failed_gps_sync_queues_[agent_id].empty()) {
      GPSmeasurement failed_gps_measurement =
          failed_gps_sync_queues_[agent_id].front();
      OdomMeasurementQueue failed_close_odoms =
          getOdomMeasurements(failed_gps_measurement.timestamp - 0.5,
                              failed_gps_measurement.timestamp + 0.5, agent_id);
      if (!close_odoms.empty()) {
        syncAndAlignGPS(agent_id, failed_gps_measurement, failed_close_odoms);
        failed_gps_sync_queues_[agent_id].pop_front();
      }
    }
  }
}

void System::fusedPclConsumerLoop(const uint64_t agent_id) {
  comm_msgs::fused_pclConstPtr fused_pcl_msg_raw, fused_pcl_msg;
  for (;;) {
    if (fused_pcl_msgs_received_[agent_id]->PopBlocking(&fused_pcl_msg_raw) ==
        false) {
      return;
    }
    fused_pcl_msgs_buffer_[agent_id].push_back(fused_pcl_msg_raw);
    if (fused_pcl_msgs_buffer_[agent_id].size() < 3) {
      continue;
    }
    while (fused_pcl_msgs_buffer_[agent_id].size() >= 3) {
      fused_pcl_msg = fused_pcl_msgs_buffer_[agent_id].front();
      fused_pcl_msgs_buffer_[agent_id].pop_front();
      uint64_t kf_anchor_id_num = fused_pcl_msg->anchorId;
      const double timestamp = fused_pcl_msg->header.stamp.toSec();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::fromROSMsg(fused_pcl_msg->fusedPointcloud, *fused_pcl_cloud);

      // pointcloud is delived in the camera frame of the anchor kf
      Identifier kf_anchor_id = std::make_pair(agent_id, kf_anchor_id_num);
      std::shared_ptr<KeyFrame> anchor_kf;

      // Hack in order to avoid throwing away recent point clouds....
      int counter = 0;
      while (anchor_kf == nullptr) {
        anchor_kf = maps_[agent_id]->getKeyFrame(kf_anchor_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ++counter;
        if (counter >= 5) {
          break;
        }
      }
      //    std::cout << "counter_" << agent_id << ": " << counter << std::endl;
      if (counter >= 5) {
        continue;
      }

      // store the pointcloud in the anchor kf
      anchor_kf->addFusedPcl(fused_pcl_cloud);

      // add anchor kf id to publish queue
      pcl_pub_kf_id_queues_[agent_id].push_back(kf_anchor_id_num);

      // get anchor kf id from front of the queue waiting to be published
      uint64_t pcl_pub_kf_id = pcl_pub_kf_id_queues_[agent_id].front();

      // check if this kf id is still in the sliding window for local
      // optimization
      Map::KFvec local_opt_keyframes = maps_[agent_id]->getMostRecentN(
          1);  // parameters_.local_opt_window_size);
      std::set<uint64_t> local_opt_kf_ids;
      //    for (auto iter = local_opt_keyframes.begin(); iter !=
      //    local_opt_keyframes.end(); iter++) {
      //      std::shared_ptr<KeyFrame> local_opt_kf_i = *iter;
      //      local_opt_kf_ids.insert(local_opt_kf_i->getId().second);
      //    }

      // if it is no longer in the set, then publish that kf's pointcloud
      if (local_opt_kf_ids.count(pcl_pub_kf_id) == 0) {
        // std::cout << "KF " << pcl_pub_kf_id << " no longer in sliding window,
        // PUBLISH" << std::endl;
        Identifier pcl_pub_kf_identifier =
            std::make_pair(agent_id, pcl_pub_kf_id);
        std::shared_ptr<KeyFrame> pcl_pub_kf =
            maps_[agent_id]->getKeyFrame(pcl_pub_kf_identifier);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pub_fused_pcl_cloud =
            pcl_pub_kf->getFusedPcl();
        const double pub_timestamp = pcl_pub_kf->getTimestamp();

        Eigen::Matrix4d T_S_C = pcl_pub_kf->getExtrinsics();
        Eigen::Matrix4d T_M_S = pcl_pub_kf->getOptimizedPose();
        Eigen::Matrix4d T_W_M;

        if (maps_[agent_id]->hasValidWorldTransformation()) {
          T_W_M = maps_[agent_id]->getWorldTransformation();
        } else {
          T_W_M = Eigen::Matrix4d::Identity();
        }

        Eigen::Matrix4d T_W_C = T_W_M * T_M_S * T_S_C;

        // publish the pointcloud
        if (maps_[agent_id]->getGPSStatus()) {
          if (maps_[agent_id]->hasValidWorldTransformation()) {
            fused_pcl_callback_(agent_id, pub_timestamp, pub_fused_pcl_cloud,
                                T_W_C);
          }
        } else {
          fused_pcl_callback_(agent_id, pub_timestamp, pub_fused_pcl_cloud,
                              T_W_C);
        }

        pcl_pub_kf_id_queues_[agent_id].pop_front();
      }
    }
  }
}

void System::pclConsumerLoop(const uint64_t agent_id) {
  sensor_msgs::PointCloud2ConstPtr pcl_msg;
  for (;;) {
    if (pcl_msgs_received_[agent_id]->PopBlocking(&pcl_msg) == false) {
      return;
    }

    // Create the internal representation
    PclMeasurement pcl_meas;
    pcl_meas.timestamp = pcl_msg->header.stamp.toSec();
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl::fromROSMsg(*pcl_msg, pcl_cloud);
    const size_t num_points = pcl_cloud.points.size();
    pcl_meas.points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      Eigen::Vector3d point(pcl_cloud.points[i].x, pcl_cloud.points[i].y,
                            pcl_cloud.points[i].z);
      Eigen::Vector3d color(pcl_cloud.points[i].r, pcl_cloud.points[i].g,
                            pcl_cloud.points[i].b);
      pcl_meas.points.push_back(point);
      pcl_meas.colors.push_back(color);
    }

    // Associate this measurement with an odometry
    OdomMeasurementQueue odom_meas = getOdomMeasurements(
        pcl_meas.timestamp - 0.05, pcl_meas.timestamp + 0.05, agent_id);

    if (!odom_meas.empty()) {
      OdomPclCombined comb_meas;
      comb_meas.timestamp = odom_meas[0].timestamp;
      comb_meas.odometry = odom_meas[0];
      comb_meas.pcl = pcl_meas;
      {
        std::lock_guard<std::mutex> lock(odom_pcl_mutex_);
        odom_pcl_queues_[agent_id].push_back(comb_meas);
      }
    }
  }
}

void System::publisherLoop(const uint64_t agent_id) {
  Result result;
  for (;;) {
    if (pub_msgs_received_[agent_id]->PopBlocking(&result) == false) {
      return;
    }
    transform_callback_(result.timestamp, agent_id, result.T_M_O, result.T_W_M,
                        result.T_M_Si);
  }
}

GPSmeasurementQueue System::getGPSmeasurements(const double& start_time,
                                               const double& end_time,
                                               const uint64_t agent_id) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (gps_queues_[agent_id].empty() || end_time < start_time ||
      start_time > gps_queues_[agent_id].back().timestamp) {
    return GPSmeasurementQueue();
  }

  std::unique_lock<std::mutex> lock(gps_mutex_);
  auto first_gps = gps_queues_[agent_id].begin();
  auto last_gps = gps_queues_[agent_id].end();
  for (auto itr = gps_queues_[agent_id].begin();
       itr != gps_queues_[agent_id].end(); ++itr) {
    if (itr->timestamp <= start_time) {
      first_gps = itr;
    }

    if (itr->timestamp >= end_time) {
      last_gps = itr;
      ++last_gps;
      break;
    }
  }

  return GPSmeasurementQueue(first_gps, last_gps);
}

OdomMeasurementQueue System::getOdomMeasurements(const double& start_time,
                                                 const double& end_time,
                                                 const uint64_t agent_id) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (odom_queues_[agent_id].empty() || end_time < start_time ||
      start_time > odom_queues_[agent_id].back().timestamp) {
    return OdomMeasurementQueue();
  }

  std::lock_guard<std::mutex> lock(odom_mutex_);
  auto first_odom = odom_queues_[agent_id].begin();
  auto last_odom = odom_queues_[agent_id].end();
  for (auto itr = odom_queues_[agent_id].begin();
       itr != odom_queues_[agent_id].end(); ++itr) {
    if (itr->timestamp <= start_time) {
      first_odom = itr;
    }

    if (itr->timestamp >= end_time) {
      last_odom = itr;
      ++last_odom;
      break;
    }
  }

  return OdomMeasurementQueue(first_odom, last_odom);
}

OdomGPScombinedQueue System::getCombinedMeasurement(const double& start_time,
                                                    const double& end_time,
                                                    const uint64_t agent_id) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (odom_gps_queues_[agent_id].empty() || end_time < start_time ||
      start_time > odom_gps_queues_[agent_id].back().timestamp) {
    return OdomGPScombinedQueue();
  }

  std::lock_guard<std::mutex> lock(odom_gps_mutex_);
  auto first_comb = odom_gps_queues_[agent_id].begin();
  auto last_comb = odom_gps_queues_[agent_id].end();
  for (auto itr = odom_gps_queues_[agent_id].begin();
       itr != odom_gps_queues_[agent_id].end(); ++itr) {
    if (itr->timestamp <= start_time) {
      first_comb = itr;
    }

    if (itr->timestamp >= end_time) {
      last_comb = itr;
      ++last_comb;
      break;
    }
  }

  return OdomGPScombinedQueue(first_comb, last_comb);
}

OdomPclCombinedQueue System::getCombinedPclMeasurement(
    const double& start_time, const double& end_time, const uint64_t agent_id) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (odom_pcl_queues_[agent_id].empty() || end_time < start_time ||
      start_time > odom_pcl_queues_[agent_id].back().timestamp) {
    return OdomPclCombinedQueue();
  }

  std::lock_guard<std::mutex> lock(odom_pcl_mutex_);
  auto first_comb = odom_pcl_queues_[agent_id].begin();
  auto last_comb = odom_pcl_queues_[agent_id].end();
  for (auto itr = odom_pcl_queues_[agent_id].begin();
       itr != odom_pcl_queues_[agent_id].end(); ++itr) {
    if (itr->timestamp <= start_time) {
      first_comb = itr;
    }

    if (itr->timestamp >= end_time) {
      last_comb = itr;
      ++last_comb;
      break;
    }
  }

  return OdomPclCombinedQueue(first_comb, last_comb);
}

int System::deleteGpsMeasurements(const double& clear_until,
                                  const uint64_t agent_id) {
  std::lock_guard<std::mutex> lock(gps_mutex_);
  if (gps_queues_[agent_id].front().timestamp > clear_until) {
    return 0;
  }

  auto erase_end = gps_queues_[agent_id].begin();
  int removed = 0;
  for (auto itr = gps_queues_[agent_id].begin();
       itr != gps_queues_[agent_id].end(); ++itr) {
    erase_end = itr;
    if (itr->timestamp >= clear_until) {
      break;
    }
    ++removed;
  }

  gps_queues_[agent_id].erase(gps_queues_[agent_id].begin(), erase_end);

  return removed;
}

int System::deleteOdomMeasurements(const double& clear_until,
                                   const uint64_t agent_id) {
  std::lock_guard<std::mutex> lock(odom_mutex_);
  if (odom_queues_[agent_id].front().timestamp > clear_until) {
    return 0;
  }

  auto erase_end = odom_queues_[agent_id].begin();
  int removed = 0;
  for (auto itr = odom_queues_[agent_id].begin();
       itr != odom_queues_[agent_id].end(); ++itr) {
    erase_end = itr;
    if (itr->timestamp >= clear_until) {
      break;
    }
    ++removed;
  }

  odom_queues_[agent_id].erase(odom_queues_[agent_id].begin(), erase_end);

  return removed;
}

int System::deletePclMeasurements(const double& clear_until,
                                  const uint64_t agent_id) {
  std::lock_guard<std::mutex> lock(pcl_mutex_);
  if (pcl_queues_[agent_id].front().timestamp > clear_until) {
    return 0;
  }

  auto erase_end = pcl_queues_[agent_id].begin();
  int removed = 0;
  for (auto itr = pcl_queues_[agent_id].begin();
       itr != pcl_queues_[agent_id].end(); ++itr) {
    erase_end = itr;
    if (itr->timestamp >= clear_until) {
      break;
    }
    ++removed;
  }

  pcl_queues_[agent_id].erase(pcl_queues_[agent_id].begin(), erase_end);

  return removed;
}

int System::deleteCombinedMeasurements(const double& clear_until,
                                       const uint64_t agent_id) {
  std::lock_guard<std::mutex> lock(odom_pcl_mutex_);
  if (odom_gps_queues_[agent_id].front().timestamp > clear_until) {
    return 0;
  }

  auto erase_end = odom_gps_queues_[agent_id].begin();
  int removed = 0;
  for (auto itr = odom_gps_queues_[agent_id].begin();
       itr != odom_gps_queues_[agent_id].end(); ++itr) {
    erase_end = itr;
    if (itr->timestamp >= clear_until) {
      break;
    }
    ++removed;
  }

  odom_gps_queues_[agent_id].erase(odom_gps_queues_[agent_id].begin(),
                                   erase_end);

  return removed;
}

int System::deleteCombinedPclMeasurements(const double& clear_until,
                                          const uint64_t agent_id) {
  std::lock_guard<std::mutex> lock(odom_pcl_mutex_);
  if (odom_pcl_queues_[agent_id].front().timestamp > clear_until) {
    return 0;
  }

  auto erase_end = odom_pcl_queues_[agent_id].begin();
  int removed = 0;
  for (auto itr = odom_pcl_queues_[agent_id].begin();
       itr != odom_pcl_queues_[agent_id].end(); ++itr) {
    erase_end = itr;
    if (itr->timestamp >= clear_until) {
      break;
    }
    ++removed;
  }

  odom_pcl_queues_[agent_id].erase(odom_pcl_queues_[agent_id].begin(),
                                   erase_end);

  return removed;
}

void System::syncAndAlignGPS(const uint64_t agent_id,
                             GPSmeasurement& gps_measurement,
                             OdomMeasurementQueue& close_odoms) {
  OdomMeasurement closest_meas;
  double min_dist = std::numeric_limits<double>::max();
  for (auto itr = close_odoms.begin(); itr != close_odoms.end(); ++itr) {
    if (std::abs(itr->timestamp - gps_measurement.timestamp) < min_dist) {
      closest_meas = (*itr);
      min_dist = std::abs(itr->timestamp - gps_measurement.timestamp);
    }
  }
  // Set the height to the closest odometry height due to large fluctuations in
  // gps readings
  if (parameters_.ignore_gps_altitude) {
    gps_measurement.local_measurement(2) = closest_meas.translation(2);
  }

  OdomGPScombined combined_meas(closest_meas, gps_measurement);
  OdomGPScombinedVector init_corresp;
  {
    std::lock_guard<std::mutex> lock(odom_gps_mutex_);
    odom_gps_queues_[agent_id].push_back(combined_meas);
    odom_gps_init_queues_[agent_id].push_back(combined_meas);
    init_corresp.reserve(odom_gps_init_queues_[agent_id].size());
    for (auto itr = odom_gps_init_queues_[agent_id].begin();
         itr != odom_gps_init_queues_[agent_id].end(); ++itr) {
      init_corresp.push_back((*itr));
    }
  }

  Eigen::Matrix4d T_W_M, covariance;
  if (!maps_[agent_id]->hasValidWorldTransformation()) {
    bool could_init = Optimizer::computeGPSalignment(
        init_corresp, parameters_.gps_parameters[agent_id].offset, T_W_M,
        covariance, parameters_, odom_gps_init_queues_[agent_id]);
    if (could_init) {
      maps_[agent_id]->setWorldTransformation(T_W_M, covariance);
      trigger_init_opt_[agent_id] = true;
      ROS_INFO(
          "[PGB] Initialized the GPS reference transformation for agent %lu",
          agent_id);
    }
  }
}

void System::updatePath(const uint64_t agent_id) {
  std::lock_guard<std::mutex> lock(path_mutex_);

  paths_[agent_id].poses.clear();

  for (auto keyframe : maps_[agent_id]->getAllKeyFrames()) {
    Eigen::Matrix4d T_W_M = maps_[agent_id]->getWorldTransformation();
    Eigen::Matrix4d T_S_C = keyframe->getExtrinsics();
    Eigen::Matrix4d T_M_S = keyframe->getOptimizedPose();
    Eigen::Matrix4d T_W_C = T_W_M * T_M_S * T_S_C;
    const Eigen::Quaterniond q_W_C(T_W_C.block<3, 3>(0, 0));

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(keyframe->getTimestamp());
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = T_W_C(0, 3);
    pose_stamped.pose.position.y = T_W_C(1, 3);
    pose_stamped.pose.position.z = T_W_C(2, 3);
    pose_stamped.pose.orientation.x = q_W_C.x();
    pose_stamped.pose.orientation.y = q_W_C.y();
    pose_stamped.pose.orientation.z = q_W_C.z();
    pose_stamped.pose.orientation.w = q_W_C.w();

    paths_[agent_id].poses.push_back(pose_stamped);
    paths_[agent_id].header = pose_stamped.header;
  }
  path_callback_(paths_[agent_id], agent_id);
}

void System::savePath(std::string file_path) {
  LOG(INFO) << "saving path";
  std::vector<std::ofstream> fs;
  for (size_t i = 0; i < paths_.size(); i++) {
    boost::filesystem::path p(file_path_);
    p.append("vins_c" + std::to_string(i) + ".txt");
    fs.emplace_back(std::ofstream());
    fs[i].open(p.string());
    LOG(INFO) << p.string();
  }
  for (size_t i = 0; i < paths_.size(); i++) {
    LOG(INFO) << paths_[i].poses.size();
    for (auto const& pose : paths_[i].poses) {
      LOG(INFO) << pose.pose;
      fs[i] << std::setprecision(6) << pose.header.stamp.toSec()
            << std::setprecision(7) << " " << pose.pose.position.x << " "
            << pose.pose.position.y << " " << pose.pose.position.z << " "
            << pose.pose.orientation.x << " " << pose.pose.orientation.y << " "
            << pose.pose.orientation.z << " " << pose.pose.orientation.w
            << std::endl;
    }
  }
  for (auto& f : fs) f.close();
}

}  // namespace pgbe
