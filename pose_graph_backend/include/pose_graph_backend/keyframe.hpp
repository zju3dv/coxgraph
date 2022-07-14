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
 * keyframe.hpp
 * @brief Header file for the KeyFrame Class
 * @author: Marco Karrer
 * Created on: Aug 13, 2018
 */

#pragma once

#include <deque>
#include <memory>
#include <mutex>

#include <aslam/cameras/camera.h>
#include <comm_msgs/keyframe.h>
#include <pcl/common/transforms.h>
#include <robopt_open/common/definitions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "DBoW2/DBoW2.h"
#include "measurements.hpp"
#include "parameters.hpp"
#include "pose_graph_backend/brisk-vocabulary.hpp"
#include "typedefs.hpp"

/// \brief pgbe The main namespace of this package.
namespace pgbe {

#define FRAMEGRIDCOLS 75
#define FRAMEGRIDROWS 48

class KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  /// \brief Empty constructor.
  KeyFrame(){};
  ~KeyFrame();

  /// \brief Construct a keyframe from a keyframe message.
  /// @param keyframe_msg The keyframe message.
  /// @param params The system parameters.
  KeyFrame(const comm_msgs::keyframeConstPtr& keyframe_msg,
           const SystemParameters& params, const uint64_t agent_id);

  /// \brief Fill a keyframe with its data.
  /// @param frame_id The id of the keyframe.
  /// @param agent_id The id of the agent from which the keyframe is coming.
  /// @param keypoints The keypoint positions (distorted coordinates).
  /// @param descriptors The keypoint descriptors corresponding to the kpts.
  /// @param landmarks The associated landmarks (in camera coordinates).
  /// @param kpt_idxs The corresponding keypoint indexes of the landmarks.
  /// @param connections The frame ids for the connected KFs.
  /// @param T_R_S The odometry pose (in its local coordinate frame).
  void constructKeyFrame(const uint64_t frame_id, const uint64_t agent_id,
                         const double timestamp, const Vector2Vector& keypoints,
                         const cv::Mat& descriptors,
                         const Vector3Vector& landmarks,
                         const std::vector<size_t>& kpt_idxs,
                         const std::vector<uint64_t>& connections,
                         const Eigen::Matrix4d& T_O_S);

  /// \brief Project a point into the image.
  /// @param l_C The point in the camera coordinate system.
  /// @param proj The projected point.
  /// @return The aslam projection output.
  aslam::ProjectionResult projectPoint(const Eigen::Vector3d& l_C,
                                       Eigen::Vector2d& proj);

  /// \brief Add an odometry connection.
  /// @param frame_id The frame id of the connected frame.
  /// @return True if connection was inserted, false if it already existed.
  bool insertOdomConnection(const uint64_t frame_id);

  /// \brief Remove an odometry connection.
  /// @param frame_id The frame id of the keyframe whose connection should
  ///                 be removed.
  /// @return True if connection was removed, false if the connection did
  ///         not exist.
  bool removeOdomConnection(const uint64_t frame_id);

  /// \brief Insert a new loop closure connection.
  /// @param loop_id The full id of the loop frame.
  /// @return True if the connection was inserted, false if it exists already.
  bool insertLoopClosureConnection(const Identifier& loop_id);

  /// \brief Remove a loop closure connection.
  /// @param loop_id The full id of the loop frame.
  /// @return True if the connection was removed, false if it did not exist.
  bool removeLoopClosureConnection(const Identifier& loop_id);

  /// \brief Get the keyframes connected by odometry links.
  /// @return The identifiers of the connected keyframes.
  std::set<Identifier> getOdomConnections();

  /// \brief Get the connected keyframes.
  /// @return The identifiers of the connected keyframes.
  std::set<Identifier> getLoopConnections();

  /// \brief Insert a new loop closure connection.
  /// @param loop_edge The loop closure edge (includes the transformation).
  /// @return True if the connection was inserted, false if there exists already
  ///         an edge.
  bool insertLoopClosureEdge(const LoopEdge& loop_edge);

  /// \brief Remove a loop closure edge.
  /// @param loop_edge The loop closure edge.
  /// @return True if the connection was deleted, false if there did not exist
  ///         an edge
  bool removeLoopClosureEdge(const LoopEdge& loop_edge);

  /// \brief Get the loop-closure edges.
  /// @return The loop closure matches (incl. their transformation).
  LoopEdges getLoopClosureEdges() { return loop_edges_; }

  /// \brief Get the raw pointer to a descriptor.
  /// @param kp_idx The keypoint index.
  /// @return The pointer to the descriptor.
  inline const unsigned char* getKeypointDescriptor(size_t kp_idx) {
    return descriptors_.data + descriptors_.cols * kp_idx;
  }

  /// \brief Get the number of keypoints in this frame.
  /// @return The number of keypoints.
  size_t getNumKeypoints() { return keypoints_.size(); }

  /// \brief Get a keypoint.
  /// @param kp_idx The keypoint index.
  /// @return The distorted keypoint.
  Eigen::Vector2d getKeypoint(const size_t kp_idx);

  /// \brief Get a undistorted keypoint.
  /// @param kp_idx The keypoint index.
  /// @return The undistorted keypoint.
  Eigen::Vector3d getKeypointBearing(const size_t kp_idx);

  /// \brief Get a landmark position.
  /// @param kp_idx The keypoint index.
  /// @param landmark The landmark if available.
  /// @return Whether a landmark is associated.
  bool getLandmark(const size_t kp_idx, Eigen::Vector3d& landmark);

  /// \brief Get the keyframe id.
  /// @return The keyframe unique id.
  Identifier getId() { return id_; }

  /// \brief Get the keyframe timestamp (in s)
  /// @return The timestamp.
  double getTimestamp() { return timestamp_; }

  /// \brief Get the average focal length.
  /// @return The average focal length.
  double getFocalLength();

  /// \brief Get a pointer to the camera.
  /// @return The raw pointer to the camera.
  aslam::Camera* getCamera() { return camera_->clone(); }

  /// \brief Get the extrinsics transformation (T_S_C).
  /// @return The extrinsics transformation.
  Eigen::Matrix4d getExtrinsics();

  /// \brief Set the temporary loop pose.
  /// @param T_M_S The pose for the loop-closure optimization.
  void setLoopClosurePose(const Eigen::Matrix4d& T_M_S);

  /// \brief Get the temporary loop pose.
  /// @return The pose for the loop-closure optimization.
  Eigen::Matrix4d getLoopClosurePose() { return T_M_S_loop_tmp_; }

  /// \brief Get the odometry pose.
  /// @return The odometry pose of this keyframe.
  Eigen::Matrix4d getOdometryPose() { return T_O_S_; }

  /// \brief Set the optimized pose.
  /// @param T_M_S The pose that should be set.
  void setOptimizedPose(const Eigen::Matrix4d& T_M_S);

  /// \brief Get the optimized pose.
  /// @return The optimized pose of this keyframe.
  Eigen::Matrix4d getOptimizedPose();

  /// \brief Set the prior covariance.
  /// @param cov_T_M_S The pose covariance.
  void setOptimizedPoseCovariance(const Eigen::Matrix4d& cov_T_M_S);

  /// \brief Get the prior covariance.
  /// @param cov_T_M_S The pose covariance.
  /// @return Whether or not the covariance was actually set.
  bool getOptimizedPoseCovariance(Eigen::Matrix4d& cov_T_M_S);

  /// \brief Add GPS measurement.
  /// @param measurement The combined gps-odometry measurement.
  void addGpsMeasurement(const OdomGPScombined& measurement) {
    gps_measurements_.push_back(measurement);
  }

  /// \brief Get the GPS measurements.
  /// @return The GPS measurements associated to this keyframe.
  OdomGPScombinedVector getGpsMeasurements() { return gps_measurements_; }

  /// \brief Add point cloud message.
  /// @param pcl_msg The point cloud message.
  void addPclMessage(const sensor_msgs::PointCloud2& pcl_msg);

  /// \brief Add fused pointcloud to the keyframe
  /// \@param fused_plc_cloud Pointer to PCL format pointcloud
  void addFusedPcl(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud);

  /// \brief Get the fused pointcloud of the system
  /// \@return Pointer to PCL format pointcloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr getFusedPcl() {
    return fused_pcl_cloud_ptr;
  }

  /// \brief Get the point cloud message.
  /// @param pcl_msg The point cloud message for this frame.
  /// @return Whether or not there was a point cloud.
  bool getPclMessage(sensor_msgs::PointCloud2& pcl_msg);

  /// \brieg Write the latest loop closure transformation to csv for debug
  /// \@param filename Name of the file to write to
  /// \@param lc The keyframe from where the loop was detected
  /// \@param T_A_B The relative pose transformation from the loop closure
  void writeLoopClosureTransform(const std::string& filename,
                                 std::shared_ptr<KeyFrame> lc,
                                 Eigen::Matrix4d& T_A_B);

  /// \brief Get the image (only for debugging)
  /// @return The image.
  cv::Mat getImage() { return image_.clone(); }

  /// \brieg Get the number of landmarks
  /// \@preturn The number of landmarks
  size_t getNumLandmarks() { return landmarks_.size(); }

  // TODO: Make proper interfacing for this!
  DBoW2::BowVector bow_vec_;
  DBoW2::FeatureVector feat_vec_;
  double ceres_pose_loop_[robopt::defs::pose::kPoseBlockSize];
  double ceres_pose_[robopt::defs::pose::kPoseBlockSize];
  double ceres_extrinsics_[robopt::defs::pose::kPoseBlockSize];

 protected:
  /// \brief Compute the BoW representation.
  void computeBoW();

  /// \brief Conver the descriptors to a vector of descriptors
  std::vector<cv::Mat> toDescriptorVector();

  /// \brief Assign the features to a position in a grid.
  void assignFeaturesToGrid();

  /// \brief Get the position of a image point within the grid.
  /// @param kp The point position.
  /// @param pos_x The x-position.
  /// @param pos_y The y-position.
  void positionInGrid(const Eigen::Vector2d& kp, int& pos_x, int& pos_y);

  // Identifier
  Identifier id_;
  double timestamp_;

  // Odometry connections
  std::set<uint64_t> connections_odom_;

  // Loop closure connections
  std::set<Identifier> connections_loop_;
  LoopEdges loop_edges_;

  // Odometry Data
  Eigen::Matrix4d T_O_S_;
  Eigen::Matrix4d T_M_S_;
  Eigen::Matrix4d T_M_S_loop_tmp_;
  Eigen::Matrix4d cov_T_M_S_;
  bool has_covariance_;

  // GPS measurements
  std::vector<OdomGPScombined, Eigen::aligned_allocator<OdomGPScombined>>
      gps_measurements_;

  // The keypoint data
  Vector2Vector keypoints_;
  cv::Mat descriptors_;
  Vector3Vector landmarks_;
  std::vector<int> landmark_index_;
  cv::Mat image_;

  // The keypoint grid
  double grid_element_inv_width_;
  double grid_element_inv_height_;
  std::vector<size_t> grid_[FRAMEGRIDCOLS][FRAMEGRIDROWS];

  // Disparity Image
  bool has_point_cloud_;
  sensor_msgs::PointCloud2 pointcloud_;

  // Store the parameters
  SystemParameters parameters_;
  aslam::Camera::Ptr camera_;

  // BoW
  std::shared_ptr<BRISKVocabulary> voc_ptr_;

  // Mutex for protected data access
  std::mutex mutex_connections_;
  std::mutex mutex_pose_;
  std::mutex mutex_fn;

  // fused pointcloud (only present if keyframe is an anchor kf for a fused
  // pointcloud)
  pcl::PointCloud<pcl::PointXYZRGB> fused_pcl_cloud_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_pcl_cloud_ptr;
};

}  // namespace pgbe
