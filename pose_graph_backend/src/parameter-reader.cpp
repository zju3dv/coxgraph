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
 * parameter-reader.cpp
 * @brief Implementation file for the ParameterReader Class
 * @author: Marco Karrer
 * Created on: Aug 14, 2018
 */

#include "parameter-reader.hpp"

#include <ros/package.h>
#include <boost/filesystem.hpp>

namespace pgbe {

ParameterReader::ParameterReader(ros::NodeHandle &nh, const size_t num_agents)
    : nh_(&nh), num_agents_(num_agents), read_parameters_(false) {}

ParameterReader::~ParameterReader() {}

bool ParameterReader::getParameters(SystemParameters &params) {
  if (read_parameters_) {
    params = parameters_;
    return true;
  } else {
    return readParameters(params);
  }
}

bool ParameterReader::readParameters(SystemParameters &params) {
  CameraParametersVector cam_vector;
  cam_vector.reserve(num_agents_);
  GpsParametersVector gps_parameters;
  gps_parameters.reserve(num_agents_);
  std::vector<bool> gps_active;
  gps_active.reserve(num_agents_);
  bool successful = true;

  bool simulation_ = false;
  if (!nh_->getParam("simulation", simulation_)) {
    ROS_WARN("[PGB] Parameter 'simulation' missing");
    successful = false;
  }

  for (size_t i = 0; i < num_agents_; ++i) {
    // Get the config file for the camera
    const std::string param_name_i = "cam_config" + std::to_string(i);
    std::string file_name_i;
    if (!nh_->getParam(param_name_i, file_name_i)) {
      ROS_WARN_STREAM("[PGB] Parameter " << param_name_i << " missing");
      successful = false;
      break;
    }

    // Read the camera configuration file
    CameraParameters cam_params_i(i, file_name_i);
    if (cam_params_i.camera == NULL) {
      ROS_WARN_STREAM("[PGB] Could not read parameters for camera " << i);
      successful = false;
      break;
    }
    cam_vector.push_back(cam_params_i);

    // Read the GPS configuration
    const std::string gps_offset_name_i = "gps_offset" + std::to_string(i);
    std::vector<double> gps_offset_i;
    if (!nh_->getParam(gps_offset_name_i, gps_offset_i)) {
      ROS_WARN_STREAM("[PGB] Parameter " << gps_offset_name_i << " missing");
      successful = false;
      break;
    }
    const std::string gps_reference_name_i =
        "gps_reference" + std::to_string(i);
    std::vector<double> gps_reference_i;
    if (!nh_->getParam(gps_reference_name_i, gps_reference_i)) {
      ROS_WARN_STREAM("[PGB] Parameter " << gps_reference_name_i << " missing");
      successful = false;
      break;
    }
    gps_parameters.push_back(GpsParameters(
        i,
        Eigen::Vector3d(gps_reference_i[0], gps_reference_i[1],
                        gps_reference_i[2]),
        Eigen::Vector3d(gps_offset_i[0], gps_offset_i[1], gps_offset_i[2])));

    const std::string gps_active_name_i = "gps_active_" + std::to_string(i);
    bool gps_active_i = false;
    if (!nh_->getParam(gps_active_name_i, gps_active_i)) {
      ROS_WARN_STREAM("[PGB] Parameter " << gps_active_name_i << " missing");
      successful = false;
    }
    gps_active.push_back(gps_active_i);
  }

  // Read loop detection performance parameters
  double loop_candidate_min_score;
  if (!nh_->getParam("loop_candidate_min_score", loop_candidate_min_score)) {
    ROS_WARN("[PGB] Parameter 'loop_candidate_min_score' missing");
    successful = false;
  }

  int loop_image_min_matches;
  if (!nh_->getParam("loop_image_min_matches", loop_image_min_matches)) {
    ROS_WARN("[PGB] Parameter 'loop_image_min_matches' missing");
    successful = false;
  }

  int loop_detect_sac_thresh;
  if (!nh_->getParam("loop_detect_sac_thresh", loop_detect_sac_thresh)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_sac_thresh' missing");
    successful = false;
  }

  int loop_detect_sac_max_iter;
  if (!nh_->getParam("loop_detect_sac_max_iter", loop_detect_sac_max_iter)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_sac_max_iter' missing");
    successful = false;
  }

  int loop_detect_min_sac_inliers;
  if (!nh_->getParam("loop_detect_min_sac_inliers",
                     loop_detect_min_sac_inliers)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_min_sac_inliers' missing");
    successful = false;
  }

  int loop_detect_min_sac_inv_inliers;
  if (!nh_->getParam("loop_detect_min_sac_inv_inliers",
                     loop_detect_min_sac_inv_inliers)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_min_sac_inv_inliers' missing");
    successful = false;
  }

  int loop_detect_min_pose_inliers;
  if (!nh_->getParam("loop_detect_min_pose_inliers",
                     loop_detect_min_pose_inliers)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_min_pose_inliers' missing");
    successful = false;
  }

  double rel_pose_outlier_norm_min;
  if (!nh_->getParam("rel_pose_outlier_norm_min", rel_pose_outlier_norm_min)) {
    ROS_WARN("[PGB] Parameter 'rel_pose_outlier_norm_min' missing");
    successful = false;
  }

  double loop_detect_reset_time;
  if (!nh_->getParam("loop_detect_reset_time", loop_detect_reset_time)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_reset_time' missing");
    successful = false;
  }

  int max_loop_candidates;
  if (!nh_->getParam("max_loop_candidates", max_loop_candidates)) {
    ROS_WARN("[PGB] Parameter 'max_loop_candidates' missing");
    successful = false;
  }

  int gps_align_num_corr;
  if (!nh_->getParam("gps_align_num_corr", gps_align_num_corr)) {
    ROS_WARN("[PGB] Parameter 'gps_align_num_corr' missing");
    successful = false;
  }

  double gps_align_cov_max;
  if (!nh_->getParam("gps_align_cov_max", gps_align_cov_max)) {
    ROS_WARN("[PGB] Parameter 'gps_align_cov_max' missing");
    successful = false;
  }

  int loop_detect_skip_kf;
  if (!nh_->getParam("loop_detect_skip_kf", loop_detect_skip_kf)) {
    ROS_WARN("[PGB] Parameter 'loop_detect_skip_kf' missing");
    successful = false;
  }

  double information_odom_drift_yaw;
  if (!nh_->getParam("information_odom_drift_yaw",
                     information_odom_drift_yaw)) {
    ROS_WARN("[PGB] Parameter 'information_odom_drift_yaw' missing");
    successful = false;
  }

  double information_odom_drift_p;
  if (!nh_->getParam("information_odom_drift_p", information_odom_drift_p)) {
    ROS_WARN("[PGB] Parameter 'information_odom_drift_p' missing");
    successful = false;
  }

  double information_odom_map_yaw;
  if (!nh_->getParam("information_odom_map_yaw", information_odom_map_yaw)) {
    ROS_WARN("[PGB] Parameter 'information_odom_map_yaw' missing");
    successful = false;
  }

  double information_odom_map_p;
  if (!nh_->getParam("information_odom_map_p", information_odom_map_p)) {
    ROS_WARN("[PGB] Parameter 'information_odom_map_p' missing");
    successful = false;
  }

  double information_odom_edges_yaw;
  if (!nh_->getParam("information_odom_edges_yaw",
                     information_odom_edges_yaw)) {
    ROS_WARN("[PGB] Parameter 'information_odom_edges_yaw' missing");
    successful = false;
  }

  double information_odom_edges_p;
  if (!nh_->getParam("information_odom_edges_p", information_odom_edges_p)) {
    ROS_WARN("[PGB] Parameter 'information_odom_edges_p' missing");
    successful = false;
  }

  double information_loop_edges_yaw;
  if (!nh_->getParam("information_loop_edges_yaw",
                     information_loop_edges_yaw)) {
    ROS_WARN("[PGB] Parameter 'information_loop_edges_yaw' missing");
    successful = false;
  }

  double information_loop_edges_p;
  if (!nh_->getParam("information_loop_edges_p", information_loop_edges_p)) {
    ROS_WARN("[PGB] Parameter 'information_loop_edges_p' missing");
    successful = false;
  }

  bool ignore_gps_altitude;
  if (!nh_->getParam("ignore_gps_altitude", ignore_gps_altitude)) {
    ROS_WARN("[PGB] Parameter 'ignore_gps_altitude' missing");
    successful = false;
  }

  int local_opt_window_size;
  if (!nh_->getParam("local_opt_window_size", local_opt_window_size)) {
    ROS_WARN("[PGB] Parameter 'local_opt_window_size' missing");
    successful = false;
  }

  int rel_pose_corr_min;
  if (!nh_->getParam("rel_pose_corr_min", rel_pose_corr_min)) {
    ROS_WARN("[PGB] Parameter 'rel_pose_corr_min' missing");
    successful = false;
  }

  bool logging;
  if (!nh_->getParam("logging", logging)) {
    ROS_WARN("[PGB] Parameter 'logging' missing");
    successful = false;
  }

  std::string log_folder("");
  if (logging) {
    // Get the log folder -- by default, the estimations will be dumped in a log
    // folder in the pose_graph_backend package
    // To avoid overwriting, use the current time stamp as the name of the
    // folder Visit http://en.cppreference.com/w/cpp/chrono/c/strftime for more
    // information about date/time format
    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char buf[80];
    strftime(buf, sizeof(buf), "/logs/%Y-%m-%d_%X", &tstruct);

    log_folder = ros::package::getPath("pose_graph_backend") + buf;
    ROS_INFO_STREAM("[PGB] Logging folder: " << log_folder);

    // If the log folder does not exist, then create it
    boost::filesystem::path log_folder_path(log_folder);
    if (!boost::filesystem::exists(log_folder_path)) {
      if (!boost::filesystem::create_directories(log_folder_path)) {
        ROS_ERROR("[PGB] Could not create log folders - Abort...");
        return -1;
      }
    }
  }

  if (successful) {
    params = SystemParameters(
        num_agents_, simulation_, cam_vector, gps_parameters,
        loop_candidate_min_score, loop_image_min_matches,
        loop_detect_sac_thresh, loop_detect_sac_max_iter,
        loop_detect_min_sac_inliers, loop_detect_min_sac_inv_inliers,
        loop_detect_min_pose_inliers, rel_pose_outlier_norm_min,
        loop_detect_reset_time, max_loop_candidates, gps_align_num_corr,
        gps_align_cov_max, gps_active, loop_detect_skip_kf,
        information_odom_drift_yaw, information_odom_drift_p,
        information_odom_map_yaw, information_odom_map_p,
        information_odom_edges_yaw, information_odom_edges_p,
        information_loop_edges_yaw, information_loop_edges_p,
        ignore_gps_altitude, local_opt_window_size, rel_pose_corr_min, logging,
        log_folder);
  } else {
    return successful;
  }

  // Read the vocabulary
  std::string bow_voc_file;
  successful &= nh_->getParam("bow_voc", bow_voc_file);
  params.voc_ptr = std::make_shared<BRISKVocabulary>(bow_voc_file);

  return successful;
}

}  // namespace pgbe
