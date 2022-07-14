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
 * system.hpp
 * @brief Header file for the System Class
 * @author: Marco Karrer
 * Created on: Aug 14, 2018
 */

#pragma once

#include <comm_msgs/fused_pcl.h>
#include <comm_msgs/keyframe.h>
#include <coxgraph_mod/vio_interface.h>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_srvs/Empty.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <geodetic_utils/geodetic_conv.hpp>
#include <memory>
#include <mutex>
#include <thread>

#include "measurements.hpp"
#include "parameters.hpp"
#include "pose_graph_backend/keyframe-database.hpp"
#include "pose_graph_backend/keyframe.hpp"
#include "pose_graph_backend/loop-detection.hpp"
#include "pose_graph_backend/map.hpp"
#include "threadsafe/ThreadsafeQueue.hpp"

/// \brief pgbe The main namespace of this package.
namespace pgbe {

struct KeyFrameFull {
  comm_msgs::keyframeConstPtr keyframe;
  sensor_msgs::PointCloud2::ConstPtr pointcloud;
};

class System {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Useful typedefs for this class
  typedef okvis::threadsafe::ThreadSafeQueue<comm_msgs::keyframeConstPtr>
      KeyFrameMsgQueue;
  typedef std::shared_ptr<KeyFrameMsgQueue> KeyFrameMsgQueuePtr;
  typedef okvis::threadsafe::ThreadSafeQueue<nav_msgs::OdometryConstPtr>
      OdometryMsgQueue;
  typedef std::shared_ptr<OdometryMsgQueue> OdometryMsgQueuePtr;
  typedef okvis::threadsafe::ThreadSafeQueue<sensor_msgs::PointCloud2ConstPtr>
      PclMsgQueue;
  typedef std::shared_ptr<PclMsgQueue> PclMsgQueuePtr;
  typedef okvis::threadsafe::ThreadSafeQueue<sensor_msgs::NavSatFixConstPtr>
      GpsMsgQueue;
  typedef std::shared_ptr<GpsMsgQueue> GpsMsgQueuePtr;
  typedef okvis::threadsafe::ThreadSafeQueue<std::shared_ptr<KeyFrame>>
      KeyFrameQueue;
  typedef std::shared_ptr<KeyFrameQueue> KeyFrameQueuePtr;
  typedef okvis::threadsafe::ThreadSafeQueue<Result> ResultQueue;
  typedef std::shared_ptr<ResultQueue> ResultQueuePtr;

  typedef okvis::threadsafe::ThreadSafeQueue<comm_msgs::fused_pclConstPtr>
      FusedPclMsgQueue;
  typedef std::shared_ptr<FusedPclMsgQueue> FusedPclMsgQueuePtr;
  typedef std::deque<comm_msgs::fused_pclConstPtr> FusedPclMsgDequeue;

  // Callback typedefs (for publishing)
  typedef std::function<void(double t, const uint64_t agent_id,
                             const Result& result)>
      FullCallback;

  typedef std::function<void(double t, const uint64_t agent_id,
                             const Eigen::Matrix4d&, const Eigen::Matrix4d&,
                             const Eigen::Matrix4d&)>
      TransformCallback;

  typedef std::function<void(nav_msgs::Path& msg_path, const uint64_t agent_id)>
      PathCallback;

  typedef std::function<void(const uint64_t agent_id, const double& timestamp,
                             const Eigen::Matrix4d T_W_C)>
      CamVizCallback;

  typedef std::function<void(
      const uint64_t agent_id, const double& timestamp,
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud,
      const Eigen::Matrix4d& T_W_C)>
      FusedPCLCallback;

  /// \brief Empty constructor
  System() {}
  ~System();

  /// \brief Constructor
  /// @param params The system parameters.
  System(const SystemParameters& params, const ros::NodeHandle& nh,
         const ros::NodeHandle& nh_private);

  /// \brief Set the transform callback.
  /// @param callback The callback function.
  void setTransformCallback(const TransformCallback& callback) {
    transform_callback_ = callback;
  }

  /// \brief Set the transform callback.
  /// @param callback The callback function.
  void setPCLCallback(const FullCallback& callback) {
    pcl_callback_ = callback;
  }

  /// \brief Set the Path visualization callback.
  /// @param callback The callback function.
  void setPathCallback(const PathCallback& callback) {
    path_callback_ = callback;
  }

  /// \brief Set the camera visualization callback.
  /// @param callback The callback function.
  void setCamVizCallback(const CamVizCallback& callback) {
    cam_viz_callback_ = callback;
  }

  /// \brief Set the fused pointcloud callback.
  /// @param callback The callback function.
  void setFusedPCLCallback(const FusedPCLCallback& callback) {
    fused_pcl_callback_ = callback;
  }

  /// \brief Add a new keyframe message to be processed.
  /// @param keyframe_msg The keyframe message with the associated pointcloud.
  void addKeyFrameMsg(const comm_msgs::keyframeConstPtr& keyframe_msg,
                      const uint64_t agent_id);

  /// \brief Add a new odometry message.
  /// @param odom_msg The odometry message.
  /// @param agent_id The id from which agent it came from.
  void addOdometryMsg(const nav_msgs::OdometryConstPtr& keyframe_msg,
                      const uint64_t agent_id);

  /// \brief Add a new point cloud message.
  /// @param pcl_msg The point cloud message.
  /// @param agent_id The id from which agent it came from.
  void addPointCloudMsg(const sensor_msgs::PointCloud2ConstPtr& pcl_msg,
                        const uint64_t agent_id);

  /// \brief Add a new gps measurement to be processed.
  /// @param gps_msg The new gps message.
  /// @param agent_id The id from which agent it came from.
  void addGpsMsg(const sensor_msgs::NavSatFixConstPtr& gps_msg,
                 const uint64_t agent_id);

  /// \brief Add a new fused pointcloud message to be stored.
  /// @param fused_pcl_msg The new fused pointcloud message.
  /// @param agent_id The id from which agent it came from.
  void addFusedPointCloudMsg(const comm_msgs::fused_pclConstPtr& fused_pcl_msg,
                             const uint64_t agent_id);

  /// \brief Add a new fused pointcloud message to be stored.
  /// @param gps_measurement The gps_measurement we are attempting to sync and
  /// align.
  /// @param agent_id The id from which agent it came from.
  /// @param close_odoms The close odometry measurements that we can sync GPS to
  void syncAndAlignGPS(const uint64_t agent_id, GPSmeasurement& gps_measurement,
                       OdomMeasurementQueue& close_odoms);

 protected:
  /// \brief Initialize the system and start the threads
  void init();

  /// \brief Keyframe measurement consumer loop.
  void keyframeConsumerLoop(const uint64_t agent_id);

  /// \brief Perform the keyframe optimization.
  void optimizerLoop(const uint64_t agent_id);

  /// \brief GPS measurement consumer loop.
  void gpsConsumerLoop(const uint64_t agent_id);

  /// \brief PCL measurement consumer loop.
  void pclConsumerLoop(const uint64_t agent_id);

  /// \brief Publisher loop.
  void publisherLoop(const uint64_t agent_id);

  void fusedPclConsumerLoop(const uint64_t agent_id);

  /// \brief Get all GPS measurements within a time horizon.
  /// @param start_time The lower time boarder.
  /// @param end_time The upper time boarder.
  /// @param agent_id The id for which it is requested.
  /// @return The GPS measurements queue, empty if no measurements.
  GPSmeasurementQueue getGPSmeasurements(const double& start_time,
                                         const double& end_time,
                                         const uint64_t agent_id);

  /// \brief Get all odometry measurements within a time horizon.
  /// @param start_time The lower time boarder.
  /// @param end_time The upper time boarder.
  /// @param agent_id The id for which it is requested.
  /// @return The Odometry measurements queue, empty if no measurements.
  OdomMeasurementQueue getOdomMeasurements(const double& start_time,
                                           const double& end_time,
                                           const uint64_t agent_id);

  /// \brief Get all combined GPS/odom measurements within a time horizon.
  /// @param start_time The lower time boarder.
  /// @param end_time The upper time boarder.
  /// @param agent_id The id for which it is requested.
  /// @return The combined measurements queue, empty if no measurements.
  OdomGPScombinedQueue getCombinedMeasurement(const double& start_time,
                                              const double& end_time,
                                              const uint64_t agent_id);

  /// \brief Get all combined PCL/odom measurements within a time horizon.
  /// @param start_time The lower time boarder.
  /// @param end_time The upper time boarder.
  /// @param agent_id The id for which it is requested.
  /// @return The combined measurements queue, empty if no measurements.
  OdomPclCombinedQueue getCombinedPclMeasurement(const double& start_time,
                                                 const double& end_time,
                                                 const uint64_t agent_id);

  /// \brief Clear all GPS measurements until the given time.
  /// @param clear_until The timestamp.
  /// @param agent_id The id for which the action should be performed.
  /// @return The number of cleared measurements.
  int deleteGpsMeasurements(const double& clear_until, const uint64_t agent_id);

  /// \brief Clear all odometry measurements until the given time.
  /// @param clear_until The timestamp.
  /// @param agent_id The id for which the action should be performed.
  /// @return The number of cleared measurements.
  int deleteOdomMeasurements(const double& clear_until,
                             const uint64_t agent_id);

  /// \brief Clear all pcl measurements until the given time.
  /// @param clear_until The timestamp.
  /// @param agent_id The id for which the action should be performed.
  /// @return The number of cleared measurements.
  int deletePclMeasurements(const double& clear_until, const uint64_t agent_id);

  /// \brief Clear all combined measurements until the given time.
  /// @param clear_until The timestamp.
  /// @param agent_id The id for which the action should be performed.
  /// @return The number of cleared measurements.
  int deleteCombinedMeasurements(const double& clear_until,
                                 const uint64_t agent_id);

  /// \brief Clear all combined measurements until the given time.
  /// @param clear_until The timestamp.
  /// @param agent_id The id for which the action should be performed.
  /// @return The number of cleared measurements.
  int deleteCombinedPclMeasurements(const double& clear_until,
                                    const uint64_t agent_id);

  /// \brief Update the agent's path for visualization
  /// @param agent_id The id of the agent to update the path
  void updatePath(const uint64_t agent_id);

  void savePath(std::string file_path);

  ros::NodeHandle nh_private_;
  ros::ServiceServer save_path_srv_;
  std::string file_path_;
  bool savePathCallback(std_srvs::Empty::Request& request,
                        std_srvs::Empty::Response& response) {
    savePath(file_path_);
    return true;
  }

  // store the path messages
  std::vector<nav_msgs::Path> paths_;

  // Store the system parameters
  SystemParameters parameters_;

  // Store the keyframe database
  std::shared_ptr<KeyFrameDatabase> database_;

  // Store the maps
  std::vector<std::shared_ptr<Map>,
              Eigen::aligned_allocator<std::shared_ptr<Map>>>
      maps_;

  // Only for initialization: Store whether or not the initial optimization
  // should be triggered
  std::vector<double> trigger_init_opt_;

  // Store the loop-detectors
  std::vector<std::shared_ptr<LoopDetection>,
              Eigen::aligned_allocator<std::shared_ptr<LoopDetection>>>
      loop_detectors_;
  std::vector<double> last_loop_closure_;

  // GPS-to local plane conversion
  std::vector<std::shared_ptr<geodetic_converter::GeodeticConverter>>
      gps_converters_;

  // Measurement preparation threads
  std::vector<std::thread> keyframe_consumer_threads_;
  std::vector<std::thread> keyframe_optimizer_threads_;
  std::vector<std::thread> gps_consumer_threads_;
  std::vector<std::thread> depth_consumer_threads_;
  std::vector<std::thread> pcl_consumrer_threads_;
  std::vector<std::thread> publisher_threads_;
  std::vector<std::thread> fused_pcl_consumer_threads_;
  // std::thread publisher_thread_;

  // Input measurement queues
  std::vector<KeyFrameMsgQueuePtr> keyframe_msgs_received_;
  std::vector<OdometryMsgQueuePtr> odom_msgs_received_;
  std::vector<PclMsgQueuePtr> pcl_msgs_received_;
  std::vector<GpsMsgQueuePtr> gps_msgs_received_;
  std::vector<ResultQueuePtr> pub_msgs_received_;

  std::vector<FusedPclMsgQueuePtr> fused_pcl_msgs_received_;
  std::vector<FusedPclMsgDequeue> fused_pcl_msgs_buffer_;
  // ResultQueuePtr pub_msgs_received_;

  // Processed queues
  std::vector<KeyFrameQueuePtr> keyframes_received_;

  // Temporary Queue to allow delayed processing of keyframes
  std::vector<std::deque<std::shared_ptr<KeyFrame>,
                         Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>>
      keyframe_buffers_;

  // Queues for raw measurements
  std::vector<OdomMeasurementQueue> odom_queues_;
  std::vector<GPSmeasurementQueue> gps_queues_;
  std::vector<OdomGPScombinedQueue> odom_gps_queues_;
  std::vector<OdomGPScombinedQueue> odom_gps_init_queues_;
  std::vector<PclMeasurementQueue> pcl_queues_;
  std::vector<OdomPclCombinedQueue> odom_pcl_queues_;

  std::vector<std::deque<uint64_t>> pcl_pub_kf_id_queues_;
  std::vector<GPSmeasurementQueue> failed_gps_sync_queues_;

  // Access mutex
  std::mutex kf_mutex_;
  std::mutex odom_mutex_;
  std::mutex pcl_mutex_;
  std::mutex gps_mutex_;
  std::mutex odom_gps_mutex_;
  std::mutex odom_pcl_mutex_;
  std::mutex opt_mutex_;
  std::mutex path_mutex_;

  // Flags to coordinate local and global optimization
  std::vector<bool*> optimization_flags_;

  // Counter used to only process loop detections for every other keyframe
  int kf_loop_detection_skip_;

  // Store the callbacks
  FullCallback pcl_callback_;
  TransformCallback transform_callback_;
  PathCallback path_callback_;
  CamVizCallback cam_viz_callback_;
  FusedPCLCallback fused_pcl_callback_;

  coxgraph::mod::VIOInterface* vio_interface_;
};

}  // namespace pgbe
