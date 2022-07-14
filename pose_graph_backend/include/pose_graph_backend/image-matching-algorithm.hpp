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
 * image-matching-algorithm.hpp
 * @brief Header file for the ImageMatchingAlgorithm Class
 * @author: Marco Karrer
 * Created on: Aug 16, 2018
 */

#pragma once

#include <brisk/internal/hamming.h>
#include <memory>

#include "matcher/DenseMatcher.hpp"
#include "matcher/MatchingAlgorithm.hpp"

#include "pose_graph_backend/keyframe.hpp"
#include "typedefs.hpp"

/// \brief pgbe The main namespace of this package.
namespace pgbe {

class ImageMatchingAlgorithm : public okvis::MatchingAlgorithm {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Constructor
  /// @param dist_threshold The threshold for a match to be considered valid.
  ImageMatchingAlgorithm(const float dist_threshold);

  virtual ~ImageMatchingAlgorithm(){};

  ///@brief Set which frames to match.
  ///@param kf1  The multiframe whose frames should be matched.
  ///@param kf2  ID of the frame inside multiframe to match.
  void setFrames(std::shared_ptr<KeyFrame> kf1, std::shared_ptr<KeyFrame> kf2);

  /// \brief This will be called exactly once for each call to
  /// DenseMatcher::match().
  virtual void doSetup();

  /// \brief What is the size of list A?
  virtual size_t sizeA() const;
  /// \brief What is the size of list B?
  virtual size_t sizeB() const;

  /// \brief Get the distance threshold for which matches exceeding it will not
  /// be returned as matches.
  virtual float distanceThreshold() const;

  /// \brief Set the distance threshold for which matches exceeding it will not
  /// be returned as matches.
  void setDistanceThreshold(float distanceThreshold);

  /// \brief Should we skip the item in list A? This will be called once for
  /// each item in the list
  virtual bool skipA(size_t index_A) const { return skip_A_[index_A]; }

  /// \brief Should we skip the item in list B? This will be called many times.
  virtual bool skipB(size_t index_B) const { return skip_B_[index_B]; }

  /**
   * @brief Calculate the distance between two keypoints.
   * @param indexA Index of the first keypoint.
   * @param indexB Index of the other keypoint.
   * @return Distance between the two keypoint descriptors.
   * @remark Points that absolutely don't match will return float::max.
   */
  virtual float distance(size_t index_A, size_t index_B) const {
    const float dist = static_cast<float>(specificDescriptorDistance(
        keyframe_A_->getKeypointDescriptor(index_A),
        keyframe_B_->getKeypointDescriptor(index_B)));

    if (dist < distance_threshold_) {
      if (verifyMatch(index_A, index_B)) return dist;
    }
    return std::numeric_limits<float>::max();
  }

  /// \brief Geometric verification of a match.
  bool verifyMatch(size_t indexA, size_t indexB) const;

  /// \brief A function that tells you how many times setMatching() will be
  /// called. \warning Currently not implemented to do anything.
  virtual void reserveMatches(size_t num_matches);

  /// \brief At the end of the matching step, this function is called once
  ///        for each pair of matches discovered.
  virtual void setBestMatch(size_t indexA, size_t indexB, double distance);

  /// \brief Get the number of matches.
  size_t numMatches();

  /// \brief Obtain the matches
  Matches getMatches() { return matches_; }

 protected:
  /// \name Which frames to take
  /// \{
  std::shared_ptr<KeyFrame> keyframe_A_;
  std::shared_ptr<KeyFrame> keyframe_B_;
  /// \}

  /// Distances above this threshold will not be returned as matches.
  float distance_threshold_;

  /// The number of matches.
  size_t num_matches_ = 0;

  // TODO: implement this
  /// Focal length of camera used in frame A.
  double fA_ = 0;
  /// Focal length of camera used in frame B.
  double fB_ = 0;

  /// Should keypoint[index] in frame A be skipped
  std::vector<bool> skip_A_;
  /// Should keypoint[index] in frame B be skipped
  std::vector<bool> skip_B_;

  /// Store the matches
  Matches matches_;

  /// \brief Calculates the distance between two descriptors.
  // copy from BriskDescriptor.hpp
  u_int32_t specificDescriptorDistance(const unsigned char* descriptorA,
                                       const unsigned char* descriptorB) const {
    return brisk::Hamming::PopcntofXORed(descriptorA, descriptorB, 3);
  }
};

}  // namespace pgbe
