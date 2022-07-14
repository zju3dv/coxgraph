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
 * optimizer.cpp
 * @brief Source file for the Optimizer Class
 * @author: Marco Karrer
 * Created on: Aug 17, 2018
 */
#include "pose_graph_backend/optimizer.hpp"

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion-equidistant.h>
#include <ceres/covariance.h>
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
#include <iostream>
#include <numeric>

namespace pgbe {

void Optimizer::homogenousToCeres(const Eigen::Matrix4d& T, double* ceres_ptr) {
  Eigen::Map<Eigen::Quaterniond> q(ceres_ptr);
  Eigen::Map<Eigen::Vector3d> p(ceres_ptr +
                                robopt::defs::pose::kOrientationBlockSize);

  p = T.block<3, 1>(0, 3);
  q = Eigen::Quaterniond(T.block<3, 3>(0, 0));
}

Eigen::Matrix4d Optimizer::ceresToHomogenous(double* ceres_ptr) {
  Eigen::Map<Eigen::Quaterniond> q(ceres_ptr);
  q.normalize();
  Eigen::Map<Eigen::Vector3d> p(ceres_ptr +
                                robopt::defs::pose::kOrientationBlockSize);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = q.toRotationMatrix();
  T.block<3, 1>(0, 3) = p;

  return T;
}

bool Optimizer::computeGPSalignment(
    const OdomGPScombinedVector& correspondences,
    const Eigen::Vector3d& antenna_pos, Eigen::Matrix4d& T_W_O,
    Eigen::Matrix4d& covariance, SystemParameters& params,
    OdomGPScombinedQueue& odom_gps_init_queue) {
  const int num_corr = correspondences.size();
  if (num_corr < params.gps_align_num_corr) {
    std::cout << "Not enough correspondences" << std::endl;
    return false;
  }

  // First transform the odometry points from IMU to antenna coordinates
  Vector3Vector gps_points, odom_points;
  gps_points.reserve(num_corr);
  odom_points.reserve(num_corr);
  for (size_t i = 0; i < num_corr; ++i) {
    gps_points.push_back(correspondences[i].gps.local_measurement);
    Eigen::Quaterniond q_O_S = correspondences[i].odometry.rotation;
    Eigen::Vector3d t_O_S = correspondences[i].odometry.translation;
    Eigen::Vector3d t_O_A = q_O_S * antenna_pos + t_O_S;
    odom_points.push_back(t_O_A);
  }

  // Create the least squares problem
  Eigen::MatrixXd M(3 * num_corr, 5);
  M.setZero();
  Eigen::VectorXd b(3 * num_corr);
  b.setZero();
  for (size_t i = 0; i < num_corr; ++i) {
    M(i * 3, 0) = odom_points[i](0);
    M(i * 3, 1) = -odom_points[i](1);
    M(i * 3, 2) = 1.0;
    b(i * 3) = gps_points[i](0);

    M(i * 3 + 1, 0) = odom_points[i](1);
    M(i * 3 + 1, 1) = odom_points[i](0);
    M(i * 3 + 1, 3) = 1.0;
    b(i * 3 + 1) = gps_points[i](1);

    M(i * 3 + 2, 4) = 1.0;
    b(i * 3 + 2) = gps_points[i](2) - odom_points[i](2);
  }
  Eigen::MatrixXd H = M.transpose() * M;
  Eigen::VectorXd h = M.transpose() * b;
  Eigen::VectorXd x = H.inverse() * h;

  if (x[0] < -1.0) {
    x[0] = -1.0;
  } else if (x[0] > 1.0) {
    x[0] = 1.0;
  }

  if (x[1] < -1.0) {
    x[1] = -1.0;
  } else if (x[1] > 1.0) {
    x[1] = 1.0;
  }
  double yaw_cos = std::acos(x(0));
  double yaw_sin = std::asin(x(1));

  if (yaw_cos * yaw_sin < 0.0) {
    yaw_cos *= -1.0;
  }

  double yaw = robopt::common::yaw::normalizeYaw((yaw_cos + yaw_sin) / 2.0);
  if (std::isnan(yaw)) {
    // get rid of old measurements that could be contributing to the NaN result
    std::cout << "yaw NaN" << std::endl;
    odom_gps_init_queue.pop_front();
    return false;
  }
  const Eigen::Quaterniond q = robopt::common::yaw::ExpMap(yaw);
  const Eigen::Vector3d p(x(2), x(3), x(4));
  T_W_O = Eigen::Matrix4d::Identity();
  T_W_O.block<3, 3>(0, 0) = q.toRotationMatrix();
  T_W_O.block<3, 1>(0, 3) = p;

  // Now try to optimize this problem!
  // Setup the ceres problem
  ceres::Problem::Options prob_opts;
  ceres::Problem problem(prob_opts);
  ceres::LocalParameterization* local_pose_yaw_parameterization =
      new robopt::local_param::PoseQuaternionYawLocalParameterization();

  // Add the only variable
  double ceres_T_W_O[robopt::defs::pose::kPoseBlockSize];
  homogenousToCeres(T_W_O, ceres_T_W_O);
  problem.AddParameterBlock(ceres_T_W_O, robopt::defs::pose::kPoseBlockSize,
                            local_pose_yaw_parameterization);

  // Create all the pose variables (--> will all be set to constant)
  double ceres_antenna[robopt::defs::pose::kPositionBlockSize];
  ceres_antenna[0] = antenna_pos(0);
  ceres_antenna[1] = antenna_pos(1);
  ceres_antenna[2] = antenna_pos(2);
  problem.AddParameterBlock(ceres_antenna,
                            robopt::defs::pose::kPositionBlockSize);
  problem.SetParameterBlockConstant(ceres_antenna);
  double ceres_poses[num_corr][robopt::defs::pose::kPoseBlockSize];
  Eigen::Quaterniond q_rel;
  q_rel.setIdentity();
  Eigen::Vector3d t_rel;
  t_rel.setZero();
  for (size_t i = 0; i < num_corr; ++i) {
    // Add the variable and set it constant
    Eigen::Quaterniond q_O_Si = correspondences[i].odometry.rotation;
    Eigen::Matrix4d T_O_Si = Eigen::Matrix4d::Identity();
    T_O_Si.block<3, 3>(0, 0) = q_O_Si.toRotationMatrix();
    T_O_Si.block<3, 1>(0, 3) = correspondences[i].odometry.translation;
    homogenousToCeres(T_O_Si, ceres_poses[i]);
    problem.AddParameterBlock(ceres_poses[i],
                              robopt::defs::pose::kPoseBlockSize,
                              local_pose_yaw_parameterization);
    problem.SetParameterBlockConstant(ceres_poses[i]);

    // Create a residual block
    ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
        robopt::posegraph::GpsErrorAutoDiff, 3, 7, 7, 3>(
        new robopt::posegraph::GpsErrorAutoDiff(
            gps_points[i], q_rel, t_rel, correspondences[i].gps.covariance));
    problem.AddResidualBlock(f, NULL, ceres_T_W_O, ceres_poses[i],
                             ceres_antenna);
  }

  // Optimize the problem
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 5;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << std::endl;

  // Recover the covariance of the transformation
  ceres::Covariance::Options cov_options;
  ceres::Covariance ceres_covariance(cov_options);
  std::vector<const double*> covariance_blocks;
  covariance_blocks.push_back(ceres_T_W_O);
  CHECK(ceres_covariance.Compute(covariance_blocks, &problem));
  double ceres_covariance_min[16];
  ceres_covariance.GetCovarianceMatrixInTangentSpace(covariance_blocks,
                                                     ceres_covariance_min);
  Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> cov_min(
      ceres_covariance_min);
  covariance = cov_min;
  T_W_O = ceresToHomogenous(ceres_T_W_O);
  std::cout << "GPS covariance "
            << ": " << std::sqrt(cov_min(0, 0)) << std::endl;
  if (std::sqrt(cov_min(0, 0)) < params.gps_align_cov_max) {
    return true;
  } else {
    return false;
  }
}

Eigen::Matrix4d Optimizer::computeMapTransformation(
    const Map::KFvec& keyframes, const Eigen::Matrix4d& T_M_O_init,
    SystemParameters& system_parameters) {
  // We want to use the relative pose constraint, so set the world frame to
  // identity and the odometry frame to T_W_O. With setting the world frame
  // constant we only optimize the relative pose.
  double ceres_map[robopt::defs::pose::kPoseBlockSize];
  double ceres_odom[robopt::defs::pose::kPoseBlockSize];
  double ceres_dummy1[robopt::defs::pose::kPoseBlockSize];
  double ceres_dummy2[robopt::defs::pose::kPoseBlockSize];
  homogenousToCeres(Eigen::Matrix4d::Identity(), ceres_map);
  homogenousToCeres(T_M_O_init, ceres_odom);
  homogenousToCeres(Eigen::Matrix4d::Identity(), ceres_dummy1);
  homogenousToCeres(Eigen::Matrix4d::Identity(), ceres_dummy2);

  // Setup the ceres problem
  ceres::Problem::Options prob_opts;
  ceres::Problem problem(prob_opts);
  ceres::LocalParameterization* local_parameterization =
      new robopt::local_param::PoseQuaternionYawLocalParameterization();

  // Insert the variables
  problem.AddParameterBlock(ceres_map, robopt::defs::pose::kPoseBlockSize,
                            local_parameterization);
  problem.SetParameterBlockConstant(ceres_map);
  problem.AddParameterBlock(ceres_odom, robopt::defs::pose::kPoseBlockSize,
                            local_parameterization);
  problem.AddParameterBlock(ceres_dummy1, robopt::defs::pose::kPoseBlockSize,
                            local_parameterization);
  problem.SetParameterBlockConstant(ceres_dummy1);
  problem.AddParameterBlock(ceres_dummy2, robopt::defs::pose::kPoseBlockSize,
                            local_parameterization);
  problem.SetParameterBlockConstant(ceres_dummy2);

  // Add the measurements
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
    const Eigen::Matrix4d T_M_Si = keyframe_i->getOptimizedPose();
    const Eigen::Quaterniond q_M_Si(T_M_Si.block<3, 3>(0, 0));
    const Eigen::Matrix4d T_O_Si = keyframe_i->getOdometryPose();
    const Eigen::Quaterniond q_O_Si(T_O_Si.block<3, 3>(0, 0));
    const Eigen::Matrix4d T_O_M = T_O_Si * T_M_Si.inverse();
    const Eigen::Quaterniond q_O_M(T_O_M.block<3, 3>(0, 0));
    const Eigen::Vector3d p_meas = T_O_M.block<3, 1>(0, 3);
    const double yaw_meas =  // robopt::common::yaw::normalizeYaw(
        0.0 - robopt::common::yaw::LogMap(q_O_M.inverse());  //);
    Eigen::Matrix4d information;
    information.setZero();
    information(0, 0) = system_parameters.information_odom_map_yaw;
    information(1, 1) = information(2, 2) = information(3, 3) =
        system_parameters.information_odom_map_p;
    ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
        robopt::posegraph::FourDofBetweenErrorAutoDiff, 4, 7, 7, 7, 7>(
        new robopt::posegraph::FourDofBetweenErrorAutoDiff(
            yaw_meas, p_meas, information,
            robopt::defs::pose::PoseErrorType::kImu));
    problem.AddResidualBlock(f, NULL, ceres_map, ceres_odom, ceres_dummy1,
                             ceres_dummy2);
  }
  Eigen::Matrix<double, 4, 4> cov_T_M_O;
  cov_T_M_O.setIdentity();
  cov_T_M_O(0, 0) = 1.0 / system_parameters.information_odom_drift_yaw;
  cov_T_M_O(1, 1) = cov_T_M_O(2, 2) = cov_T_M_O(3, 3) =
      1.0 / system_parameters.information_odom_drift_p;
  const Eigen::Quaterniond q_M_O(T_M_O_init.block<3, 3>(0, 0));
  const double yaw_M_O = robopt::common::yaw::LogMap(q_M_O);
  ceres::CostFunction* odom_prior =
      new ceres::AutoDiffCostFunction<robopt::posegraph::FourDofPriorAutoDiff,
                                      4, 7>(
          new robopt::posegraph::FourDofPriorAutoDiff(
              yaw_M_O, T_M_O_init.block<3, 1>(0, 3), cov_T_M_O));
  problem.AddResidualBlock(odom_prior, NULL, ceres_odom);

  // Solve the problem
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << std::endl;

  // Return the solution
  return ceresToHomogenous(ceres_odom);
  // return Eigen::Matrix4d::Identity();
}

int Optimizer::optimizeRelativePose(std::shared_ptr<KeyFrame> keyframe_A,
                                    std::shared_ptr<KeyFrame> keyframe_B,
                                    const Matches& landmarks_from_B_in_A,
                                    const Matches& landmarks_from_A_in_B,
                                    Eigen::Matrix4d& T_A_B,
                                    SystemParameters& params) {
  // Setup the ceres problem
  ceres::Problem::Options prob_opts;
  prob_opts.enable_fast_removal = true;
  ceres::Problem problem(prob_opts);
  ceres::LossFunction* loss_function;
  loss_function = new ceres::CauchyLoss(1.0);
  ceres::LocalParameterization* local_pose_parameterization =
      new robopt::local_param::PoseQuaternionLocalParameterization();

  // Transform the relative transformation into array format
  double ceres_A_B[robopt::defs::pose::kPoseBlockSize];
  Eigen::Map<Eigen::Quaterniond> q_A_B(ceres_A_B);
  Eigen::Map<Eigen::Vector3d> p_A_B(
      ceres_A_B + robopt::defs::visual::kOrientationBlockSize);
  q_A_B = Eigen::Quaterniond(T_A_B.block<3, 3>(0, 0));
  p_A_B = T_A_B.block<3, 1>(0, 3);
  problem.AddParameterBlock(ceres_A_B, robopt::defs::pose::kPoseBlockSize,
                            local_pose_parameterization);

  // Get the camera objects
  aslam::Camera* cam_ptr_A = keyframe_A->getCamera();
  aslam::Camera* cam_ptr_B = keyframe_B->getCamera();

  // Store the residual indexes for fast outlier removal
  size_t num_terms =
      landmarks_from_A_in_B.size() + landmarks_from_B_in_A.size();
  std::vector<ceres::ResidualBlockId> resid_ids;
  resid_ids.reserve(num_terms);
  int num_correspondences = 0;

  // Add the connections for the first frame.
  for (size_t i = 0; i < landmarks_from_B_in_A.size(); ++i) {
    // Get the keypoint measurement
    Eigen::Vector2d measurement =
        keyframe_A->getKeypoint(landmarks_from_B_in_A[i].idx_B);

    // Get the landmark
    Eigen::Vector3d landmark;
    if (!keyframe_B->getLandmark(landmarks_from_B_in_A[i].idx_A, landmark)) {
      // ROS_ERROR("Landmark from B requested that does not exist!!");
      continue;
    }

    // Setup the residual block
    ceres::CostFunction* rel_err_i =
        new robopt::reprojection::RelativeEuclideanReprError<
            aslam::Camera, aslam::EquidistantDistortion>(
            measurement, 2.0, cam_ptr_A, landmark,
            robopt::defs::visual::RelativeProjectionType::kNormal);
    ceres::ResidualBlockId tmp_id =
        problem.AddResidualBlock(rel_err_i, loss_function, ceres_A_B);
    resid_ids.push_back(tmp_id);
    ++num_correspondences;
  }

  // Add the connections for the second frame.
  for (size_t i = 0; i < landmarks_from_A_in_B.size(); ++i) {
    // Get the keypoint measurement
    Eigen::Vector2d measurement =
        keyframe_B->getKeypoint(landmarks_from_A_in_B[i].idx_B);

    // Get the landmark
    Eigen::Vector3d landmark;
    if (!keyframe_A->getLandmark(landmarks_from_A_in_B[i].idx_A, landmark)) {
      // ROS_ERROR("Landmark from A requested that does not exist!!");
      continue;
    }

    // Setup the residual block
    ceres::CostFunction* rel_err_i =
        new robopt::reprojection::RelativeEuclideanReprError<
            aslam::Camera, aslam::EquidistantDistortion>(
            measurement, 2.0, cam_ptr_B, landmark,
            robopt::defs::visual::RelativeProjectionType::kInverse);
    ceres::ResidualBlockId tmp_id =
        problem.AddResidualBlock(rel_err_i, loss_function, ceres_A_B);
    resid_ids.push_back(tmp_id);
    ++num_correspondences;
  }

  // Solve
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout <<"Relative Pose Optimization Report: " << std::endl;
  // std::cout << summary.FullReport() << std::endl;

  // Check for outliers
  ceres::Problem::EvaluateOptions eval_opts;
  eval_opts.residual_blocks = resid_ids;
  double total_cost = 0.0;
  std::vector<double> residuals;
  problem.Evaluate(eval_opts, &total_cost, &residuals, NULL, NULL);
  size_t resid_ind = 0;
  size_t num_bad = 0;
  for (size_t i = 0; i < resid_ids.size(); ++i) {
    Eigen::Vector2d resid(residuals[resid_ind], residuals[resid_ind + 1]);
    resid_ind += 2;
    std::cout << resid.norm() << std::endl;
    if (resid.norm() > params.rel_pose_outlier_norm_min) {
      problem.RemoveResidualBlock(resid_ids[i]);
      ++num_bad;
    }
  }
  std::cout << "Num correspondences: " << num_correspondences
            << "\t Num bad: " << num_bad << std::endl;
  // Perfor outlier "free" optimization
  if (num_correspondences - num_bad < params.rel_pose_corr_min) {
    return 0;
  }

  options.max_num_iterations = 50;
  ceres::Solve(options, &problem, &summary);
  // std::cout <<"Outlier Free Optimization Report: " << std::endl;
  // std::cout << summary.FullReport() << std::endl;

  // Recover the transformation
  T_A_B.block<3, 3>(0, 0) = q_A_B.toRotationMatrix();
  T_A_B.block<3, 1>(0, 3) = p_A_B;

  return num_correspondences - num_bad;
}

void Optimizer::optimizeMapPoseGraph(const MapVec& maps_ptr, bool* stop_flag,
                                     SystemParameters& system_parameters) {
  // Setup the ceres problem
  ceres::Problem::Options prob_opts;
  ceres::Problem problem(prob_opts);
  ceres::LossFunction* loss_function;
  loss_function = new ceres::CauchyLoss(1.0);
  ceres::LocalParameterization* local_pose_yaw_parameterization =
      new robopt::local_param::PoseQuaternionYawLocalParameterization();
  ceres::LocalParameterization* local_pose_parameterization =
      new robopt::local_param::PoseQuaternionLocalParameterization();

  // Get the number of maps currently merged
  const size_t num_maps = maps_ptr.size();

  // Extract keyframes and add map dependent variables
  Map::KFvec keyframes;
  std::vector<Map::KFvec> recent_keyframes;
  recent_keyframes.resize(num_maps);
  double ceres_world_map[num_maps][robopt::defs::pose::kPoseBlockSize];
  double ceres_antenna[num_maps][robopt::defs::pose::kPositionBlockSize];
  std::map<uint64_t, bool> add_gps;
  std::map<uint64_t, bool> has_gps_map;
  std::map<uint64_t, size_t> agent_to_idx;
  bool has_gps = false;

  // Add gps related variables and error terms
  for (size_t i = 0; i < num_maps; ++i) {
    // Extract the keyframes for this map
    agent_to_idx.insert(std::make_pair(maps_ptr[i]->getAgentId(), i));
    Map::KFvec keyframes_i = maps_ptr[i]->getAllKeyFrames();
    keyframes.insert(keyframes.begin(), keyframes_i.begin(), keyframes_i.end());
    recent_keyframes[i] = maps_ptr[i]->getMostRecentN(10);
    if (recent_keyframes[i].empty()) {
      std::cout << "No KeyFrames for Map: " << i << std::endl;
      return;
    }

    // does agent have a valid world transformation available for optimization
    const bool add_gps_i = maps_ptr[i]->hasValidWorldTransformation() &&
                           system_parameters.gps_active[i];
    add_gps.insert(std::make_pair(maps_ptr[i]->getAgentId(), add_gps_i));

    // does the agent have a GPS signal
    has_gps = maps_ptr[i]->getGPSStatus();
    has_gps_map.insert(
        std::make_pair(maps_ptr[i]->getAgentId(), maps_ptr[i]->getGPSStatus()));

    if (add_gps_i) {
      // has_gps = true;
      Eigen::Matrix4d T_W_M_i = maps_ptr[i]->getWorldTransformation();
      homogenousToCeres(T_W_M_i, ceres_world_map[i]);
      Eigen::Matrix4d cov_T_W_M_i = maps_ptr[i]->getWorldTransformationCov();
      problem.AddParameterBlock(ceres_world_map[i],
                                robopt::defs::pose::kPoseBlockSize,
                                local_pose_yaw_parameterization);

      if (maps_ptr[i]->getWorldAnchor()) {
        problem.SetParameterBlockConstant(ceres_world_map[i]);
        std::cout << "Agent " << maps_ptr[i]->getAgentId() << " set as anchor"
                  << std::endl;
      }

      if (has_gps) {
        Eigen::Vector3d gps_antenna = maps_ptr[i]->getGpsAntennaPosition();
        ceres_antenna[i][0] = gps_antenna(0);
        ceres_antenna[i][1] = gps_antenna(1);
        ceres_antenna[i][2] = gps_antenna(2);
        problem.AddParameterBlock(ceres_antenna[i],
                                  robopt::defs::pose::kPositionBlockSize);
        problem.SetParameterBlockConstant(ceres_antenna[i]);
      }

      // Add a prior on transformation
      const Eigen::Quaterniond q_W_M_i(T_W_M_i.block<3, 3>(0, 0));
      const double yaw_W_M_i = robopt::common::yaw::LogMap(q_W_M_i);
      ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
          robopt::posegraph::FourDofPriorAutoDiff, 4, 7>(
          new robopt::posegraph::FourDofPriorAutoDiff(
              yaw_W_M_i, T_W_M_i.block<3, 1>(0, 3), cov_T_W_M_i));
      problem.AddResidualBlock(f, NULL, ceres_world_map[i]);
    }
  }

  // Add and initialize the keyframe variables
  std::cout << "Num KFs to optimize over: " << keyframes.size() << std::endl;
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];

    // Initialize the ceres parameters
    homogenousToCeres(keyframe_i->getLoopClosurePose(),
                      keyframe_i->ceres_pose_loop_);
    homogenousToCeres(keyframe_i->getExtrinsics(),
                      keyframe_i->ceres_extrinsics_);

    problem.AddParameterBlock(keyframe_i->ceres_pose_loop_,
                              robopt::defs::pose::kPoseBlockSize,
                              local_pose_yaw_parameterization);
    problem.AddParameterBlock(keyframe_i->ceres_extrinsics_,
                              robopt::defs::pose::kPoseBlockSize,
                              local_pose_parameterization);
    problem.SetParameterBlockConstant(keyframe_i->ceres_extrinsics_);

    // Only need to add Odometry/GPS measurements for those agents with GPS
    if (has_gps_map[keyframe_i->getId().first]) {
      const size_t map_vec_idx = agent_to_idx[keyframe_i->getId().first];
      Eigen::Matrix4d T_O_Si = keyframe_i->getOdometryPose();
      OdomGPScombinedVector gps_meas = keyframe_i->getGpsMeasurements();
      if (gps_meas.empty()) {
        ROS_WARN("Keyframe does not have any associated GPS!");
        continue;
      }

      for (size_t k = 0; k < gps_meas.size(); ++k) {
        OdomGPScombined meas_k = gps_meas[k];
        Eigen::Matrix4d T_O_Sk = Eigen::Matrix4d::Identity();
        T_O_Sk.block<3, 3>(0, 0) = meas_k.odometry.rotation.toRotationMatrix();
        T_O_Sk.block<3, 1>(0, 3) = meas_k.odometry.translation;
        Eigen::Matrix4d T_rel = T_O_Si.inverse() * T_O_Sk;
        Eigen::Quaterniond q_rel(T_rel.block<3, 3>(0, 0));
        // Create a residual block
        ceres::CostFunction* f =
            new ceres::AutoDiffCostFunction<robopt::posegraph::GpsErrorAutoDiff,
                                            3, 7, 7, 3>(
                new robopt::posegraph::GpsErrorAutoDiff(
                    meas_k.gps.local_measurement, q_rel,
                    T_rel.block<3, 1>(0, 3), meas_k.gps.covariance));
        problem.AddResidualBlock(f, NULL, ceres_world_map[map_vec_idx],
                                 keyframe_i->ceres_pose_loop_,
                                 ceres_antenna[map_vec_idx]);
      }
    }
  }
  std::set<std::pair<Identifier, Identifier>> inserted_edges;

  //////////////////////
  // Insert the ESTABLISH loop closures - currently not implemented, but system
  // could be updated to add LC extra verification and store in loopConnections
  // instead of keeping all loop closures stored in loop closure edges object
  //////////////////////
  // for (size_t i = 0; i < keyframes.size(); ++i) {
  //   std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
  //   std::set<Identifier> connections = keyframe_i->getLoopConnections();

  //   // Loop over all established loop connections and insert the error terms
  //   for (auto itr = connections.begin(); itr != connections.end(); ++itr) {
  //     const uint64_t agent_id = itr->first;
  //     std::pair<Identifier, Identifier> id_pair;

  //     if (agent_id < keyframe_i->getId().first) {
  //       id_pair.first = (*itr);
  //       id_pair.second = keyframe_i->getId();
  //     } else {
  //       id_pair.first = keyframe_i->getId();
  //       id_pair.second = (*itr);
  //     }

  //     if (inserted_edges.count(id_pair)) {
  //       continue;
  //     }

  //     // Compute the measurement for this pair
  //     size_t adj_agent_id = agent_to_idx.find(agent_id)->second;
  //     std::shared_ptr<KeyFrame> keyframe_k =
  //         maps_ptr[adj_agent_id]->getKeyFrame((*itr));

  //     if (!keyframe_k) {
  //       //ROS_ERROR("Requested KeyFrame does not exist");
  //       continue;
  //     }

  //     if (keyframe_k->getId() == keyframe_i->getId()) {
  //       std::cout << "Established Loop has duplicated KFs" << std::endl;
  //       continue;
  //     }

  //     //////////////////////////////////////////

  //     inserted_edges.insert(id_pair);

  //     if (keyframe_i->getId().first == keyframe_k->getId().first) {
  //       const Eigen::Matrix4d T_M_Si = keyframe_i->getOptimizedPose();
  //       const Eigen::Matrix4d T_M_Sk = keyframe_k->getOptimizedPose();
  //       const Eigen::Matrix4d T_Sk_Si = T_M_Sk.inverse() * T_M_Si;
  //       const Eigen::Quaterniond q_M_Si(T_M_Si.block<3,3>(0, 0));
  //       const Eigen::Quaterniond q_M_Sk(T_M_Sk.block<3,3>(0, 0));
  //       const Eigen::Vector3d p_meas = T_Sk_Si.block<3,1>(0, 3);
  //       Eigen::Matrix4d information;
  //       information.setZero();
  //       information(0, 0) = system_parameters.information_loop_edges_yaw;
  //       information(1, 1) = information(2, 2) = information(3, 3) =
  //       system_parameters.information_loop_edges_p; const double yaw_Si_Sk =
  //       robopt::common::yaw::LogMap(q_M_Si) -
  //           robopt::common::yaw::LogMap(q_M_Sk);
  //       ceres::CostFunction* f
  //           = new ceres::AutoDiffCostFunction<
  //          << robopt::posegraph::FourDofBetweenErrorAutoDiff, 4, 7, 7, 7, 7>(
  //             new robopt::posegraph::FourDofBetweenErrorAutoDiff(
  //               yaw_Si_Sk, p_meas, information,
  //               robopt::defs::pose::PoseErrorType::kImu));
  //       //      robopt::posegraph::FourDofBetweenError* f = new
  //       //          robopt::posegraph::FourDofBetweenError(yaw_Si_Sk, p_meas,
  //       //            information, robopt::defs::pose::PoseErrorType::kImu);
  //       problem.AddResidualBlock(f, NULL, keyframe_i->ceres_pose_loop_,
  //           keyframe_k->ceres_pose_loop_, keyframe_i->ceres_extrinsics_,
  //           keyframe_k->ceres_extrinsics_);
  //     } else {
  //       const size_t map_idx_i = agent_to_idx[keyframe_i->getId().first];
  //       const size_t map_idx_k = agent_to_idx[keyframe_k->getId().first];
  //       const Eigen::Matrix4d T_M_Si = keyframe_i->getOptimizedPose();
  //       const Eigen::Matrix4d T_M_Sk = keyframe_k->getOptimizedPose();
  //       const Eigen::Matrix4d T_W_Mi = maps_ptr[map_idx_i]->
  //           getWorldTransformation();
  //       const Eigen::Matrix4d T_W_Mk = maps_ptr[map_idx_k]->
  //           getWorldTransformation();
  //       const Eigen::Matrix4d T_W_Si = T_W_Mi * T_M_Si;
  //       const Eigen::Matrix4d T_W_Sk = T_W_Mk * T_M_Sk;
  //       const Eigen::Matrix4d T_Sk_Si = T_W_Sk.inverse() * T_W_Si;
  //       const Eigen::Quaterniond q_W_Si(T_W_Si.block<3,3>(0, 0));
  //       const Eigen::Quaterniond q_W_Sk(T_W_Sk.block<3,3>(0, 0));
  //       const Eigen::Vector3d p_meas = T_Sk_Si.block<3,1>(0, 3);
  //       Eigen::Matrix4d information;
  //       information.setZero();
  //       information(0, 0) = system_parameters.information_loop_edges_yaw;
  //       information(1, 1) = information(2, 2) = information(3, 3) =
  //       system_parameters.information_loop_edges_p; const double yaw_Si_Sk =
  //       robopt::common::yaw::LogMap(q_W_Si) -
  //           robopt::common::yaw::LogMap(q_W_Sk);
  //       ceres::CostFunction* f
  //           = new ceres::AutoDiffCostFunction<
  //           robopt::posegraph::FourDofBetweenErrorAutoDiff2,
  //             4, 7, 7, 7, 7, 7, 7>(
  //             new robopt::posegraph::FourDofBetweenErrorAutoDiff2(
  //               yaw_Si_Sk, p_meas, information,
  //               robopt::defs::pose::PoseErrorType::kImu));
  //       //      robopt::posegraph::FourDofBetweenError* f = new
  //       //          robopt::posegraph::FourDofBetweenError(yaw_Si_Sk, p_meas,
  //       //            information, robopt::defs::pose::PoseErrorType::kImu);
  //       problem.AddResidualBlock(f, NULL,
  //           ceres_world_map[map_idx_i], ceres_world_map[map_idx_k],
  //           keyframe_i->ceres_pose_loop_, keyframe_k->ceres_pose_loop_,
  //           keyframe_i->ceres_extrinsics_, keyframe_k->ceres_extrinsics_);
  //     }
  //   }
  // }

  // Insert the odometry edges
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
    std::set<Identifier> connections = keyframe_i->getOdomConnections();
    // Loop over all established loop connections and insert the error terms
    for (auto itr = connections.begin(); itr != connections.end(); ++itr) {
      const uint64_t agent_id = itr->first;
      std::pair<Identifier, Identifier> id_pair;

      if (agent_id < keyframe_i->getId().first) {
        id_pair.first = (*itr);
        id_pair.second = keyframe_i->getId();
      } else {
        id_pair.first = keyframe_i->getId();
        id_pair.second = (*itr);
      }

      std::pair<Identifier, Identifier> swap_id_pair;
      swap_id_pair.first = id_pair.second;
      swap_id_pair.second = id_pair.first;

      if (inserted_edges.count(id_pair) || inserted_edges.count(swap_id_pair)) {
        continue;
      }

      // Compute the measurement for this pair
      size_t adj_agent_id = agent_to_idx.find(agent_id)->second;
      std::shared_ptr<KeyFrame> keyframe_k =
          maps_ptr[adj_agent_id]->getKeyFrame((*itr));

      if (!keyframe_k) {
        ROS_ERROR("Requested KeyFrame does not exist");
        continue;
      }

      if (keyframe_k->getId() == keyframe_i->getId()) {
        continue;
      }

      inserted_edges.insert(id_pair);

      const Eigen::Matrix4d T_O_Si = keyframe_i->getOdometryPose();
      const Eigen::Matrix4d T_O_Sk = keyframe_k->getOdometryPose();
      const Eigen::Matrix4d T_Sk_Si = T_O_Sk.inverse() * T_O_Si;
      const Eigen::Quaterniond q_O_Si(T_O_Si.block<3, 3>(0, 0));
      const Eigen::Quaterniond q_O_Sk(T_O_Sk.block<3, 3>(0, 0));
      const Eigen::Vector3d p_meas = T_Sk_Si.block<3, 1>(0, 3);
      Eigen::Matrix4d information;
      information.setZero();
      information(0, 0) = system_parameters.information_odom_edges_yaw;
      information(1, 1) = information(2, 2) = information(3, 3) =
          system_parameters.information_odom_edges_p;
      const double yaw_Si_Sk = robopt::common::yaw::LogMap(q_O_Si) -
                               robopt::common::yaw::LogMap(q_O_Sk);
      ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
          robopt::posegraph::FourDofBetweenErrorAutoDiff, 4, 7, 7, 7, 7>(
          new robopt::posegraph::FourDofBetweenErrorAutoDiff(
              yaw_Si_Sk, p_meas, information,
              robopt::defs::pose::PoseErrorType::kImu));
      problem.AddResidualBlock(
          f, NULL, keyframe_i->ceres_pose_loop_, keyframe_k->ceres_pose_loop_,
          keyframe_i->ceres_extrinsics_, keyframe_k->ceres_extrinsics_);
    }
  }
  int num_residuals_odom = problem.NumResidualBlocks();

  // Finally add the new loop closure poses
  bool has_fixed_node = false;
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
    LoopEdges loop_edges = keyframe_i->getLoopClosureEdges();
    if (loop_edges.empty()) {
      continue;
    }

    for (size_t k = 0; k < loop_edges.size(); ++k) {
      LoopEdge edge_k = loop_edges[k];
      // Make sure that keyframe_i is the keyframe A in the edge
      if (edge_k.id_A != keyframe_i->getId()) {
        if (edge_k.id_B != keyframe_i->getId()) {
          continue;
        } else {
          edge_k.id_B = edge_k.id_A;
          edge_k.id_A = keyframe_i->getId();
          edge_k.T_A_B = edge_k.T_A_B.inverse();
        }
      }

      std::pair<Identifier, Identifier> id_pair;

      if (edge_k.id_B.first < keyframe_i->getId().first) {
        id_pair.first = edge_k.id_B;
        id_pair.second = keyframe_i->getId();
      } else {
        id_pair.first = keyframe_i->getId();
        id_pair.second = edge_k.id_B;
      }

      // Compute the measurement for this pair
      size_t adj_edge_agent_id = agent_to_idx.find(edge_k.id_B.first)->second;
      std::shared_ptr<KeyFrame> keyframe_k =
          maps_ptr[adj_edge_agent_id]->getKeyFrame(edge_k.id_B);

      if (!keyframe_k) {
        ROS_ERROR("Requested Loop KeyFrame does not exist");
        continue;
      }

      if (keyframe_k->getId() == keyframe_i->getId()) {
        continue;
      }

      inserted_edges.insert(id_pair);

      if (id_pair.first.first == id_pair.second.first) {
        const Eigen::Matrix4d T_M_Sk = keyframe_k->getOptimizedPose();
        const Eigen::Matrix4d T_M_Si_corr =
            keyframe_k->getOptimizedPose() * edge_k.T_A_B.inverse();
        keyframe_i->setLoopClosurePose(T_M_Si_corr);
        const Eigen::Matrix4d T_Sk_Si = T_M_Sk.inverse() * T_M_Si_corr;
        const Eigen::Quaterniond q_M_Si(T_M_Si_corr.block<3, 3>(0, 0));
        const Eigen::Quaterniond q_M_Sk(T_M_Sk.block<3, 3>(0, 0));
        const Eigen::Vector3d p_meas = T_Sk_Si.block<3, 1>(0, 3);
        Eigen::Matrix4d information;
        information.setZero();
        information(0, 0) = system_parameters.information_loop_edges_yaw;
        information(1, 1) = information(2, 2) = information(3, 3) =
            system_parameters.information_loop_edges_p;
        const double yaw_Si_Sk = robopt::common::yaw::LogMap(q_M_Si) -
                                 robopt::common::yaw::LogMap(q_M_Sk);
        ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
            robopt::posegraph::FourDofBetweenErrorAutoDiff, 4, 7, 7, 7, 7>(
            new robopt::posegraph::FourDofBetweenErrorAutoDiff(
                yaw_Si_Sk, p_meas, information,
                robopt::defs::pose::PoseErrorType::kImu));
        problem.AddResidualBlock(
            f, NULL, keyframe_i->ceres_pose_loop_, keyframe_k->ceres_pose_loop_,
            keyframe_i->ceres_extrinsics_, keyframe_k->ceres_extrinsics_);
      } else {
        const size_t map_idx_i = agent_to_idx.find(id_pair.first.first)->second;
        const size_t map_idx_k =
            agent_to_idx.find(id_pair.second.first)->second;
        const Eigen::Matrix4d T_M_Si_corr =
            keyframe_k->getOptimizedPose() * edge_k.T_A_B.inverse();
        const Eigen::Matrix4d T_M_Sk = keyframe_k->getOptimizedPose();
        const Eigen::Matrix4d T_W_Mi =
            maps_ptr[map_idx_i]->getWorldTransformation();
        const Eigen::Matrix4d T_W_Mk =
            maps_ptr[map_idx_k]->getWorldTransformation();
        const Eigen::Matrix4d T_W_Si = T_W_Mi * T_M_Si_corr;
        const Eigen::Matrix4d T_W_Sk = T_W_Mk * T_M_Sk;
        const Eigen::Matrix4d T_Sk_Si = T_W_Sk.inverse() * T_W_Si;
        const Eigen::Quaterniond q_W_Si(T_W_Si.block<3, 3>(0, 0));
        const Eigen::Quaterniond q_W_Sk(T_W_Sk.block<3, 3>(0, 0));
        const Eigen::Vector3d p_meas = T_Sk_Si.block<3, 1>(0, 3);
        Eigen::Matrix4d information;
        information.setZero();
        information(0, 0) = system_parameters.information_loop_edges_yaw;
        information(1, 1) = information(2, 2) = information(3, 3) =
            system_parameters.information_loop_edges_p;
        const double yaw_Si_Sk = robopt::common::yaw::LogMap(q_W_Si) -
                                 robopt::common::yaw::LogMap(q_W_Sk);
        ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
            robopt::posegraph::FourDofBetweenErrorAutoDiff2, 4, 7, 7, 7, 7, 7,
            7>(new robopt::posegraph::FourDofBetweenErrorAutoDiff2(
            yaw_Si_Sk, p_meas, information,
            robopt::defs::pose::PoseErrorType::kImu));
        problem.AddResidualBlock(
            f, NULL, ceres_world_map[map_idx_i], ceres_world_map[map_idx_k],
            keyframe_i->ceres_pose_loop_, keyframe_k->ceres_pose_loop_,
            keyframe_i->ceres_extrinsics_, keyframe_k->ceres_extrinsics_);
      }

      has_gps = (add_gps[keyframe_i->getId().first]) ||
                (add_gps[keyframe_k->getId().first]) || has_gps;
      if (!has_fixed_node && !has_gps) {
        problem.SetParameterBlockConstant(keyframe_k->ceres_pose_loop_);
        has_fixed_node = true;
      }
    }
  }

  // If there is no fixed node, fix the first keyframe
  if (!has_fixed_node && !has_gps) {
    problem.SetParameterBlockConstant(keyframes[0]->ceres_pose_loop_);
    std::cout << "FIXED A NODE IN THE GLOBAL OPTIMIZATION" << std::endl;
  }

  int num_residuals = problem.NumResidualBlocks();

  // Perform the optimization
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 100;
  ceres::Solver::Summary summary;

  auto start1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  auto end1 = chrono::steady_clock::now();
  std::cout
      << "Solving took: "
      << chrono::duration_cast<chrono::milliseconds>(end1 - start1).count()
      << " ms" << std::endl;
  std::cout << summary.FullReport() << std::endl;

  // If gps is added --> update the world to reference transform
  *stop_flag = true;

  for (size_t i = 0; i < num_maps; ++i) {
    if (!add_gps[i]) {
      continue;
    }
    const size_t num_kfs = recent_keyframes[i].size();
    ceres::Covariance::Options cov_options;
    ceres::Covariance ceres_covariance(cov_options);
    std::vector<const double*> covariance_blocks;
    covariance_blocks.push_back(ceres_world_map[i]);

    // TODO: Why should it be the same for both cases?
    if (num_kfs < 10) {
      covariance_blocks.push_back(
          recent_keyframes[i][num_kfs - 1]->ceres_pose_loop_);
    } else {
      covariance_blocks.push_back(
          recent_keyframes[i][num_kfs - 1]->ceres_pose_loop_);
    }

    auto start = chrono::steady_clock::now();
    bool cov_check_failed =
        !ceres_covariance.Compute(covariance_blocks, &problem);
    auto end = chrono::steady_clock::now();

    std::cout
        << "Covariance computation took: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;
    if (cov_check_failed) {
      std::cout << "FAILED COVARIANCE CHECK" << std::endl;
      continue;
    }
    double ceres_covariance_min[64];
    ceres_covariance.GetCovarianceMatrixInTangentSpace(covariance_blocks,
                                                       ceres_covariance_min);
    Eigen::Map<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> cov_min(
        ceres_covariance_min);
    const Eigen::Matrix4d covariance_T_W_M = cov_min.block<4, 4>(0, 0);
    const Eigen::Matrix4d covariance_T_M_S = cov_min.block<4, 4>(4, 4);
    const Eigen::Matrix4d T_W_M = ceresToHomogenous(ceres_world_map[i]);
    maps_ptr[i]->setWorldTransformation(T_W_M, covariance_T_W_M);

    if (num_kfs < 10) {
      recent_keyframes[i][num_kfs - 1]->setOptimizedPoseCovariance(
          covariance_T_M_S);
    } else {
      recent_keyframes[i][num_kfs - 2]->setOptimizedPoseCovariance(
          covariance_T_M_S);
    }
  }

  // Write back the optimized poses
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
    const Eigen::Matrix4d T_M_Si =
        ceresToHomogenous(keyframe_i->ceres_pose_loop_);
    keyframe_i->setOptimizedPose(T_M_Si);
  }

  // Compute the new transformation between map and odometry
  Map::KFvec keyframes_after;
  for (size_t i = 0; i < num_maps; ++i) {
    Map::KFvec keyframes_after_i = maps_ptr[i]->getAllKeyFrames();
    keyframes_after.insert(keyframes_after.begin(), keyframes_after_i.begin(),
                           keyframes_after_i.end());
    const Eigen::Matrix4d T_M_O_init = maps_ptr[i]->getOdomToMap();
    const Eigen::Matrix4d T_M_O_new = computeMapTransformation(
        recent_keyframes[i], T_M_O_init, system_parameters);
    maps_ptr[i]->setOdomToMap(T_M_O_new);
  }

  if (keyframes_after.size() != keyframes.size()) {
    // A new keyframe was inserted in the map during optimization, hence
    // we need to correct its pose
    std::sort(keyframes.begin(), keyframes.end());
    std::sort(keyframes_after.begin(), keyframes_after.end());
    Map::KFvec unopt_keyframes;
    std::set_difference(
        keyframes_after.begin(), keyframes_after.end(), keyframes.begin(),
        keyframes.end(),
        std::inserter(unopt_keyframes, unopt_keyframes.begin()));

    // Transform the keyframes
    for (size_t i = 0; i < unopt_keyframes.size(); ++i) {
      std::shared_ptr<KeyFrame> keyframe_i = unopt_keyframes[i];
      const Eigen::Matrix4d T_O_Si = keyframe_i->getOdometryPose();
      const size_t agent_id = keyframe_i->getId().first;
      const size_t adj_agent_id = agent_to_idx.find(agent_id)->second;
      const Eigen::Matrix4d T_M_O_new = maps_ptr[adj_agent_id]->getOdomToMap();
      const Eigen::Matrix4d T_M_Si = T_M_O_new * T_O_Si;
      keyframe_i->setOptimizedPose(T_M_Si);
    }
  }

  *stop_flag = false;
}

void Optimizer::optimizeLocalPoseGraph(std::shared_ptr<Map> map_ptr,
                                       const size_t& window_size,
                                       bool* stop_flag,
                                       SystemParameters& system_parameters) {
  // Return if currently a global optimization was done
  if (*stop_flag) {
    return;
  }
  // This only makes sense if there are any GPS measurements
  Map::KFvec keyframes = map_ptr->getMostRecentN(window_size);

  // Setup the ceres problem
  ceres::Problem::Options prob_opts;
  ceres::Problem problem(prob_opts);
  ceres::LossFunction* loss_function;
  loss_function = new ceres::CauchyLoss(1.0);
  ceres::LocalParameterization* local_pose_yaw_parameterization =
      new robopt::local_param::PoseQuaternionYawLocalParameterization();
  ceres::LocalParameterization* local_pose_parameterization =
      new robopt::local_param::PoseQuaternionLocalParameterization();

  // Add the parameter blocks
  const bool add_gps =
      map_ptr->hasValidWorldTransformation() && map_ptr->getGPSStatus();
  double ceres_antenna[robopt::defs::pose::kPositionBlockSize];
  double ceres_world_map[robopt::defs::pose::kPoseBlockSize];
  Eigen::Matrix4d cov_T_W_M;
  if (add_gps) {
    // Add the antenna transformation
    const Eigen::Vector3d antenna_pos = map_ptr->getGpsAntennaPosition();
    ceres_antenna[0] = antenna_pos(0);
    ceres_antenna[1] = antenna_pos(1);
    ceres_antenna[2] = antenna_pos(2);
    problem.AddParameterBlock(ceres_antenna,
                              robopt::defs::pose::kPositionBlockSize);
    problem.SetParameterBlockConstant(ceres_antenna);

    // Add the World to Reference (GPS) transformation and add prior
    const Eigen::Matrix4d T_W_M = map_ptr->getWorldTransformation();
    homogenousToCeres(T_W_M, ceres_world_map);
    problem.AddParameterBlock(ceres_world_map,
                              robopt::defs::pose::kPoseBlockSize,
                              local_pose_yaw_parameterization);
    cov_T_W_M = map_ptr->getWorldTransformationCov();
    const Eigen::Quaterniond q_W_M(T_W_M.block<3, 3>(0, 0));
    const double yaw_W_M = robopt::common::yaw::LogMap(q_W_M);
    ceres::CostFunction* f =
        new ceres::AutoDiffCostFunction<robopt::posegraph::FourDofPriorAutoDiff,
                                        4, 7>(
            new robopt::posegraph::FourDofPriorAutoDiff(
                yaw_W_M, T_W_M.block<3, 1>(0, 3), cov_T_W_M));
    problem.AddResidualBlock(f, NULL, ceres_world_map);
  }

  // Add the keyframes and if available the GPS measurements
  const size_t num_kfs = keyframes.size();
  size_t num_gps = 0;
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
    Eigen::Matrix4d T_M_Si = keyframe_i->getOptimizedPose();
    homogenousToCeres(T_M_Si, keyframe_i->ceres_pose_);
    Eigen::Matrix4d T_S_Ci = keyframe_i->getExtrinsics();
    homogenousToCeres(T_S_Ci, keyframe_i->ceres_extrinsics_);
    problem.AddParameterBlock(keyframe_i->ceres_pose_,
                              robopt::defs::pose::kPoseBlockSize,
                              local_pose_yaw_parameterization);
    problem.AddParameterBlock(keyframe_i->ceres_extrinsics_,
                              robopt::defs::pose::kPoseBlockSize,
                              local_pose_parameterization);
    problem.SetParameterBlockConstant(keyframe_i->ceres_extrinsics_);
    // If available add the GPS measurements
    if (add_gps) {
      OdomGPScombinedVector gps_meas = keyframe_i->getGpsMeasurements();
      num_gps += gps_meas.size();
      Eigen::Matrix4d T_O_Si = keyframe_i->getOdometryPose();
      for (size_t k = 0; k < gps_meas.size(); ++k) {
        OdomGPScombined gps_meas_k = gps_meas[k];
        Eigen::Matrix4d T_O_Sk = Eigen::Matrix4d::Identity();
        T_O_Sk.block<3, 3>(0, 0) =
            gps_meas_k.odometry.rotation.toRotationMatrix();
        T_O_Sk.block<3, 1>(0, 3) = gps_meas_k.odometry.translation;
        Eigen::Matrix4d T_rel = T_O_Si.inverse() * T_O_Sk;
        Eigen::Quaterniond q_rel(T_rel.block<3, 3>(0, 0));
        // Create a residual block
        ceres::CostFunction* f =
            new ceres::AutoDiffCostFunction<robopt::posegraph::GpsErrorAutoDiff,
                                            3, 7, 7, 3>(
                new robopt::posegraph::GpsErrorAutoDiff(
                    gps_meas_k.gps.local_measurement, q_rel,
                    T_rel.block<3, 1>(0, 3), gps_meas_k.gps.covariance));
        problem.AddResidualBlock(f, NULL, ceres_world_map,
                                 keyframe_i->ceres_pose_, ceres_antenna);
      }
    }

    if (i == num_kfs - 1) {
      Eigen::Matrix4d pose_cov;
      const double yaw_M_Si = robopt::common::yaw::LogMap(
          Eigen::Quaterniond(T_M_Si.block<3, 3>(0, 0)));
      if (false) {  //(keyframe_i->getOptimizedPoseCovariance(pose_cov)) {
        // We have a prior, insert it
        ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
            robopt::posegraph::FourDofPriorAutoDiff, 4, 7>(
            new robopt::posegraph::FourDofPriorAutoDiff(
                yaw_M_Si, T_M_Si.block<3, 1>(0, 3), pose_cov));
        problem.AddResidualBlock(f, NULL, keyframe_i->ceres_pose_);
      } else {
        // We have no prior, fake it
        pose_cov = Eigen::Matrix4d::Identity();
        pose_cov(0, 0) = 0.03;
        pose_cov(1, 1) = pose_cov(2, 2) = pose_cov(3, 3) = 0.05;
        ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
            robopt::posegraph::FourDofPriorAutoDiff, 4, 7>(
            new robopt::posegraph::FourDofPriorAutoDiff(
                yaw_M_Si, T_M_Si.block<3, 1>(0, 3), pose_cov));
        problem.AddResidualBlock(f, NULL, keyframe_i->ceres_pose_);
        //        problem.SetParameterBlockConstant(keyframe_i->ceres_pose_);
      }
    }
  }

  // Add the odometry connections
  std::set<std::pair<Identifier, Identifier>> inserted_edges;
  for (size_t i = 0; i < keyframes.size(); ++i) {
    std::shared_ptr<KeyFrame> keyframe_i = keyframes[i];
    std::set<Identifier> connections = keyframe_i->getOdomConnections();

    // Loop over all established loop connections and insert the error terms
    for (auto itr = connections.begin(); itr != connections.end(); ++itr) {
      const uint64_t agent_id = itr->first;
      std::pair<Identifier, Identifier> id_pair;

      if (agent_id < keyframe_i->getId().first) {
        id_pair.first = (*itr);
        id_pair.second = keyframe_i->getId();
      } else {
        id_pair.first = keyframe_i->getId();
        id_pair.second = (*itr);
      }

      if (inserted_edges.count(id_pair)) {
        continue;
      }

      // Compute the measurement for this pair, first check if within window
      std::shared_ptr<KeyFrame> keyframe_k = map_ptr->getKeyFrame((*itr));
      auto find_itr = std::find(keyframes.begin(), keyframes.end(), keyframe_k);
      if (!keyframe_k) {
        continue;
      }
      if (find_itr == keyframes.end()) {
        continue;
      }
      if (keyframe_k->getId() == keyframe_i->getId()) {
        continue;
      }

      inserted_edges.insert(id_pair);

      const Eigen::Matrix4d T_O_Si = keyframe_i->getOdometryPose();
      const Eigen::Matrix4d T_O_Sk = keyframe_k->getOdometryPose();
      const Eigen::Matrix4d T_Sk_Si = T_O_Sk.inverse() * T_O_Si;
      const Eigen::Quaterniond q_O_Si(T_O_Si.block<3, 3>(0, 0));
      const Eigen::Quaterniond q_O_Sk(T_O_Sk.block<3, 3>(0, 0));
      const Eigen::Vector3d p_meas = T_Sk_Si.block<3, 1>(0, 3);
      Eigen::Matrix4d information;
      information.setZero();
      information(0, 0) = system_parameters.information_odom_edges_yaw;
      information(1, 1) = information(2, 2) = information(3, 3) =
          system_parameters.information_odom_edges_p;
      const double yaw_Si_Sk = robopt::common::yaw::LogMap(q_O_Si) -
                               robopt::common::yaw::LogMap(q_O_Sk);
      ceres::CostFunction* f = new ceres::AutoDiffCostFunction<
          robopt::posegraph::FourDofBetweenErrorAutoDiff, 4, 7, 7, 7, 7>(
          new robopt::posegraph::FourDofBetweenErrorAutoDiff(
              yaw_Si_Sk, p_meas, information,
              robopt::defs::pose::PoseErrorType::kImu));
      problem.AddResidualBlock(
          f, NULL, keyframe_i->ceres_pose_, keyframe_k->ceres_pose_,
          keyframe_i->ceres_extrinsics_, keyframe_k->ceres_extrinsics_);
    }
  }

  // Optimize the graph
  if (*stop_flag) {
    return;
  }

  CeresStoppCallback* stop_callback = new CeresStoppCallback(stop_flag);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 15;
  // options.callbacks.push_back(stop_callback);
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << "Local Optimization Summary" << std::endl;
  // std::cout << summary.FullReport() << std::endl;

  if (*stop_flag) {
    return;
  }

  // Extract the covariance
  if (add_gps) {
    ceres::Covariance::Options cov_options;
    ceres::Covariance ceres_covariance(cov_options);
    std::vector<const double*> covariance_blocks;
    covariance_blocks.push_back(ceres_world_map);
    if (ceres_covariance.Compute(covariance_blocks, &problem)) {
      double ceres_covariance_min[16];
      ceres_covariance.GetCovarianceMatrixInTangentSpace(covariance_blocks,
                                                         ceres_covariance_min);
      Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> cov_min(
          ceres_covariance_min);
      const Eigen::Matrix4d covariance_world = cov_min.block<4, 4>(0, 0);
      // const Eigen::Matrix4d covariance_pose = cov_min.block<4,4>(4, 4);
      const Eigen::Matrix4d T_W_M = ceresToHomogenous(ceres_world_map);
      // keyframes[num_kfs - 2]->setOptimizedPoseCovariance(covariance_pose);
      map_ptr->setWorldTransformation(T_W_M, covariance_world);
    } else {
      ROS_WARN("Covariance computation failed");
    }
    //    }
  }

  // Compute the odmetry transformation
  Eigen::Matrix4d T_M_O_init = map_ptr->getOdomToMap();
  Eigen::Matrix4d T_M_O =
      computeMapTransformation(keyframes, T_M_O_init, system_parameters);
  map_ptr->setOdomToMap(T_M_O);
  for (size_t i = 0; i < keyframes.size(); ++i) {
    Eigen::Matrix4d T_M_Si = ceresToHomogenous(keyframes[i]->ceres_pose_);
    keyframes[i]->setOptimizedPose(T_M_Si);
  }
}

}  // namespace pgbe
