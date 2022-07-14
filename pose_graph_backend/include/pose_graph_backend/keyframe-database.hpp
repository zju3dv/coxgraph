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
 * keyframe-database.hpp
 * @brief Header for the KeyFrameDatabase class.
 * @author: Marco Karrer
 * Created on: Aug 15, 2018
 */

#pragma once

#include <algorithm>
#include <iterator>
#include <list>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "DBoW2/BowVector.h"

#include "parameters.hpp"
#include "pose_graph_backend/keyframe.hpp"

/// \brief pgbe The main namespace of this package
namespace pgbe {

class KeyFrameDatabase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::list<std::shared_ptr<KeyFrame>,
                    Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
      KFlist;
  typedef std::vector<std::shared_ptr<KeyFrame>,
                      Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
      KFvec;
  typedef std::set<std::shared_ptr<KeyFrame>,
                   std::less<std::shared_ptr<KeyFrame>>,
                   Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
      KFset;

  ~KeyFrameDatabase(){};

  /// \brief Constructor.
  /// @param params The system parameters.
  KeyFrameDatabase(const SystemParameters& params);

  /// \brief Add a new keyframe to the database.
  /// @param keyframe_ptr A pointer to the keyframe.
  void add(std::shared_ptr<KeyFrame> keyframe_ptr);

  /// \brief Remove a keyframe from the database.
  /// @param keyframe_ptr A pointer to the keyframe.
  void erase(std::shared_ptr<KeyFrame> keyframe_ptr);

  /// \brief Clear the keyframe database.
  void clear();

  /// \brief Detect a possible set of loop closure candidates.
  /// @param query The query keyframe.
  /// @param connected_kfs The keyframes connected to the query keyframe.
  /// @param min_score The minimal similarity score to be considered a
  ///                  candidate.
  /// @return The vector of possible candidates.
  KFvec detectLoopCandidates(std::shared_ptr<KeyFrame> query,
                             KFset connected_kfs, const double min_score,
                             const int max_loop_candidates);

 protected:
  // System parameters
  SystemParameters parameters_;

  // Inverted file
  std::vector<KFlist> inverted_file_;

  // Mutex
  std::mutex mutex_;
};

}  // namespace pgbe
