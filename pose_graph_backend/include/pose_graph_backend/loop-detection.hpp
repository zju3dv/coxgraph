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
 * loop-detection.hpp
 * @brief Header file for the LoopDetection Class
 * @author: Marco Karrer
 * Created on: Aug 16, 2018
 */

#pragma once

#include <coxgraph_mod/vio_interface.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <iostream>
#include <memory>
#include <opengv/sac/Ransac.hpp>

#include "matcher/DenseMatcher.hpp"
#include "opengv/absolute_pose/frame-noncentral-absolute-adapter.hpp"
#include "opengv/sac_problems/frame-absolute-pose-sac-problem.hpp"
#include "parameters.hpp"
#include "pose_graph_backend/image-matching-algorithm.hpp"
#include "pose_graph_backend/keyframe-database.hpp"
#include "pose_graph_backend/map.hpp"

/// \brief pgbe The main namespace of this package
namespace pgbe {

class LoopDetection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief Empty constructor
  LoopDetection() {}
  ~LoopDetection(){};

  /// \brief Loop detector constructor
  /// @param params The system parameters.
  /// @param map_ptr Pointer to the underlying map.
  /// @param database_ptr Pointer to the keyframe database.
  LoopDetection(const SystemParameters& params, std::shared_ptr<Map> map_ptr,
                std::shared_ptr<KeyFrameDatabase> database_ptr,
                coxgraph::mod::VIOInterface* vio_interface);

  /// \brief Add new keyframe and check for loop-closure.
  /// @param keyframe The new keyframe.
  /// @param only_insert When no loop-closure should be detected.
  /// @return Whether or not a loop was detected.
  bool addKeyframe(std::shared_ptr<KeyFrame> keyframe,
                   const bool only_insert = false);

 protected:
  /// \brief Create and save a debug image using showing the found matches.
  /// @param filename The complete filename and path where the image should
  ///         be saved to.
  /// @param keyframe_A The first keyframe.
  /// @param keyframe_B The second keyframe.
  /// @param matches The keypoint matches
  void saveMatchingImage(const std::string& filename,
                         std::shared_ptr<KeyFrame> keyframe_A,
                         std::shared_ptr<KeyFrame> keyframe_B,
                         const Matches& matches);

  // Store the system parameters
  SystemParameters parameters_;

  // Store the map and database
  std::shared_ptr<Map> map_ptr_;
  std::shared_ptr<KeyFrameDatabase> database_ptr_;

  // The matcher
  std::shared_ptr<okvis::DenseMatcher> matcher_;

  coxgraph::mod::VIOInterface* vio_interface_;
};

}  // namespace pgbe
