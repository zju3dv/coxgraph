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
 * loop-detection.cpp
 * @brief Source file for the LoopDetection Class
 * @author: Marco Karrer
 * Created on: Aug 16, 2018
 */

#include "pose_graph_backend/loop-detection.hpp"

#include <coxgraph_mod/vio_interface.h>

#include <memory>
#include <set>
#include <string>

#include "pose_graph_backend/optimizer.hpp"

namespace pgbe {

LoopDetection::LoopDetection(const SystemParameters& params,
                             std::shared_ptr<Map> map_ptr,
                             std::shared_ptr<KeyFrameDatabase> database_ptr,
                             coxgraph::mod::VIOInterface* vio_interface)
    : parameters_(params),
      map_ptr_(map_ptr),
      database_ptr_(database_ptr),
      matcher_(
          std::shared_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      vio_interface_(vio_interface) {}

bool LoopDetection::addKeyframe(std::shared_ptr<KeyFrame> keyframe,
                                const bool only_insert) {
  // If no loop should be detected, just insert the frame into database and
  // return;
  if (only_insert) {
    database_ptr_->add(keyframe);
    return false;
  }

  // Extract the connected keyframes
  std::set<Identifier> connected_kf_ids = keyframe->getOdomConnections();
  std::set<Identifier> loop_connections = keyframe->getLoopConnections();
  for (auto itr = loop_connections.begin(); itr != loop_connections.end();
       ++itr) {
    connected_kf_ids.insert((*itr));
  }

  KeyFrameDatabase::KFset connected_kfs;
  for (auto it = connected_kf_ids.begin(); it != connected_kf_ids.end(); ++it) {
    std::shared_ptr<KeyFrame> tmp_kf = map_ptr_->getKeyFrame((*it));
    if (tmp_kf != NULL) {
      connected_kfs.insert(tmp_kf);
    }
  }

  // Check for possible candidates in database
  auto start = chrono::steady_clock::now();
  KeyFrameDatabase::KFvec loop_candidates = database_ptr_->detectLoopCandidates(
      keyframe, connected_kfs, parameters_.loop_candidate_min_score,
      parameters_.max_loop_candidates);
  auto end = chrono::steady_clock::now();
  // std::cout << "detectLoopCandidates: " <<
  // chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms"
  // << std::endl;

  if (loop_candidates.empty()) {
    database_ptr_->add(keyframe);
    return false;
  }

  // We have some candidates--> check by BF matching if they are valid
  bool found_match = false;
  for (size_t i = 0; i < loop_candidates.size(); ++i) {
    ImageMatchingAlgorithm matching_algorithm(60.0f);
    matching_algorithm.setFrames(loop_candidates[i], keyframe);
    matcher_->match<ImageMatchingAlgorithm>(matching_algorithm);
    size_t num_matches = matching_algorithm.numMatches();
    Matches matches = matching_algorithm.getMatches();
    std::cout << "Num_matches: " << num_matches << std::endl;

    if (num_matches < parameters_.loop_image_min_matches) {
      continue;
    }

    // First check localization against old frammatchese
    start = chrono::steady_clock::now();
    opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(
        keyframe, loop_candidates[i], matches);
    opengv::sac::Ransac<
        opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
        sac_prob;
    std::shared_ptr<
        opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
        abs_poseproblem_ptr(
            new opengv::sac_problems::absolute_pose::
                FrameAbsolutePoseSacProblem(
                    adapter, opengv::sac_problems::absolute_pose::
                                 FrameAbsolutePoseSacProblem::Algorithm::GP3P));
    sac_prob.sac_model_ = abs_poseproblem_ptr;
    sac_prob.threshold_ = parameters_.loop_detect_sac_thresh;
    sac_prob.max_iterations_ = parameters_.loop_detect_sac_max_iter;
    sac_prob.computeModel(1);
    end = chrono::steady_clock::now();
    // std::cout << "SAC forward " << i << ": " <<
    // chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms"
    // << std::endl;

    std::cout << "sac Num_matches: " << sac_prob.inliers_.size() << std::endl;
    if (sac_prob.inliers_.size() < parameters_.loop_detect_min_sac_inliers) {
      continue;
    }

    // Now check the other way around
    start = chrono::steady_clock::now();
    Matches matches_inv;
    matches_inv.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
      Match match_i(matches[i].idx_B, matches[i].idx_A, matches[i].distance);
      matches_inv.push_back(match_i);
    }
    opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter_inv(
        loop_candidates[i], keyframe, matches_inv);
    opengv::sac::Ransac<
        opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
        sac_prob_inv;
    std::shared_ptr<
        opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
        abs_poseproblem_ptr_inv(
            new opengv::sac_problems::absolute_pose::
                FrameAbsolutePoseSacProblem(
                    adapter_inv,
                    opengv::sac_problems::absolute_pose::
                        FrameAbsolutePoseSacProblem::Algorithm::GP3P));

    sac_prob_inv.sac_model_ = abs_poseproblem_ptr_inv;
    sac_prob_inv.threshold_ = parameters_.loop_detect_sac_thresh;
    sac_prob_inv.max_iterations_ = parameters_.loop_detect_sac_max_iter;
    sac_prob_inv.computeModel(1);
    end = chrono::steady_clock::now();
    // std::cout << "SAC backwards " << i << ": " <<
    // chrono::duration_cast<chrono::milliseconds>(end - start).count() <<
    // std::endl;
    std::cout << "sac_prob_inv Num_matches: " << sac_prob_inv.inliers_.size()
              << std::endl;
    if (sac_prob_inv.inliers_.size() <
        parameters_.loop_detect_min_sac_inv_inliers) {
      continue;
    }

    // Extract the suitable matches
    Matches landmarks_from_B_in_A;
    landmarks_from_B_in_A.reserve(sac_prob.inliers_.size());
    for (size_t k = 0; k < sac_prob.inliers_.size(); ++k) {
      landmarks_from_B_in_A.push_back(matches[sac_prob.inliers_[k]]);
    }
    Matches landmarks_from_A_in_B;
    landmarks_from_A_in_B.reserve(sac_prob_inv.inliers_.size());
    for (size_t k = 0; k < sac_prob_inv.inliers_.size(); ++k) {
      landmarks_from_A_in_B.push_back(matches_inv[sac_prob_inv.inliers_[k]]);
    }

    Eigen::Matrix4d T_A_B = Eigen::Matrix4d::Identity();
    if (sac_prob.inliers_.size() <= sac_prob_inv.inliers_.size()) {
      T_A_B.block<3, 4>(0, 0) = sac_prob_inv.model_coefficients_;
    } else {
      T_A_B.block<3, 4>(0, 0) = sac_prob.model_coefficients_;
      T_A_B = T_A_B.inverse();
    }

    // Perform optimization
    int num_inl = Optimizer::optimizeRelativePose(
        keyframe, loop_candidates[i], landmarks_from_B_in_A,
        landmarks_from_A_in_B, T_A_B, parameters_);

    std::cout << "num_inl: " << num_inl << std::endl;
    if (num_inl < parameters_.loop_detect_min_pose_inliers) {
      continue;
    }

    std::cout << "Loop-closure for agent " << keyframe->getId().first
              << " with total of " << num_inl << " matches" << std::endl;
    // Correct the T_AB to be in the imu frame
    const Eigen::Matrix4d T_S_C_A = keyframe->getExtrinsics();
    const Eigen::Matrix4d T_S_C_B = loop_candidates[i]->getExtrinsics();
    T_A_B = T_S_C_A * T_A_B * T_S_C_B.inverse();

    vio_interface_->publishLoopClosure(
        keyframe->getId().first, keyframe->getTimestamp(),
        loop_candidates[i]->getId().first, loop_candidates[i]->getTimestamp(),
        T_A_B);

    // We have found a loop closure
    found_match = true;
    LoopEdge edge(keyframe->getId(), loop_candidates[i]->getId(), T_A_B);
    keyframe->insertLoopClosureEdge(edge);

    if (keyframe->getId().first != loop_candidates[i]->getId().first) {
      map_ptr_->setNewMerge(loop_candidates[i]->getId(), T_A_B);
    }

    std::string filename =
        "/home/btearle/Documents/debug/pgbe/loop_closures/lc_" +
        std::to_string(keyframe->getId().first) + "_" +
        std::to_string(loop_candidates[i]->getId().first) + "_" +
        std::to_string(keyframe->getId().second) + "_" + std::to_string(i) +
        ".csv";
    keyframe->writeLoopClosureTransform(filename, loop_candidates[i], T_A_B);
  }

  return found_match;
}

void LoopDetection::saveMatchingImage(const std::string& filename,
                                      std::shared_ptr<KeyFrame> keyframe_A,
                                      std::shared_ptr<KeyFrame> keyframe_B,
                                      const Matches& matches) {
  const cv::Mat img_A = keyframe_A->getImage();
  const cv::Mat img_B = keyframe_B->getImage();

  // allocate an image
  const size_t im_cols = img_A.cols;
  const size_t im_rows = img_A.rows;
  const size_t row_jump = im_rows;

  // Set the images
  cv::Mat outimg(2 * im_rows, im_cols, CV_8UC3);
  cv::Mat img_A_col = outimg(cv::Rect(0, 0, im_cols, im_rows));
  cv::Mat img_B_col = outimg(cv::Rect(0, row_jump, im_cols, im_rows));
  cv::cvtColor(img_A, img_A_col, cv::COLOR_GRAY2BGR);
  cv::cvtColor(img_B, img_B_col, cv::COLOR_GRAY2BGR);

  // Define colors
  cv::Scalar blue = cv::Scalar(255, 0, 0);      // blue
  cv::Scalar green = cv::Scalar(0, 255, 0);     // green
  cv::Scalar yellow = cv::Scalar(0, 255, 255);  // yellow
  cv::Scalar red = cv::Scalar(0, 0, 255);       // red
  double radius = 3.0;

  // Draw the keypoints
  std::cout << "Draw the keypoints" << std::endl;
  for (size_t i = 0; i < keyframe_A->getNumKeypoints(); ++i) {
    Eigen::Vector2d kpt_i = keyframe_A->getKeypoint(i);
    Eigen::Vector3d dummy_lm;
    if (keyframe_A->getLandmark(i, dummy_lm)) {
      cv::circle(img_A_col, cv::Point2f(kpt_i[0], kpt_i[1]), radius, green, 1,
                 cv::LINE_AA);
    } else {
      cv::circle(img_A_col, cv::Point2f(kpt_i[0], kpt_i[1]), radius, blue, 1,
                 cv::LINE_AA);
    }
  }
  for (size_t i = 0; i < keyframe_B->getNumKeypoints(); ++i) {
    Eigen::Vector2d kpt_i = keyframe_B->getKeypoint(i);
    Eigen::Vector3d dummy_lm;
    if (keyframe_B->getLandmark(i, dummy_lm)) {
      cv::circle(img_B_col, cv::Point2f(kpt_i[0], kpt_i[1]), radius, green, 1,
                 cv::LINE_AA);
    } else {
      cv::circle(img_B_col, cv::Point2f(kpt_i[0], kpt_i[1]), radius, blue, 1,
                 cv::LINE_AA);
    }
  }

  // Draw the matches
  std::cout << "Draw the matches" << std::endl;
  for (size_t i = 0; i < matches.size(); ++i) {
    Eigen::Vector2d kpt_A = keyframe_A->getKeypoint(matches[i].idx_A);
    Eigen::Vector2d kpt_B = keyframe_B->getKeypoint(matches[i].idx_B);
    Eigen::Vector3d dummy_lm;
    if (keyframe_A->getLandmark(matches[i].idx_A, dummy_lm) ||
        keyframe_B->getLandmark(matches[i].idx_B, dummy_lm)) {
      // Has a 3d correspondence
      cv::line(outimg, cv::Point2f(kpt_A[0], kpt_A[1]),
               cv::Point2f(kpt_B[0], kpt_B[1] + row_jump), green, 1, cv::LINE_AA);
    } else {
      // No 3d correspondence
      cv::line(outimg, cv::Point2f(kpt_A[0], kpt_A[1]),
               cv::Point2f(kpt_B[0], kpt_B[1] + row_jump), blue, 1, cv::LINE_AA);
    }
  }
  cv::imwrite(filename, outimg);
}

}  // namespace pgbe
