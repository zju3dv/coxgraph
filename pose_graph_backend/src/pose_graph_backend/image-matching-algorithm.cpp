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
 * image-matching-algorithm.cpp
 * @brief Source file for the ImageMatchingAlgorithm Class
 * @author: Marco Karrer
 * Created on: Aug 16, 2018
 */

#include "pose_graph_backend/image-matching-algorithm.hpp"

namespace pgbe {

// Constructor
ImageMatchingAlgorithm::ImageMatchingAlgorithm(const float distance_threshold)
    : distance_threshold_(distance_threshold) {}

void ImageMatchingAlgorithm::setFrames(std::shared_ptr<KeyFrame> kf1,
                                       std::shared_ptr<KeyFrame> kf2) {
  keyframe_A_ = kf1;
  keyframe_B_ = kf2;

  // TODO: Set the focal length
  //  fA_ = (kfPtr1->fx + kfPtr1->fy)/2.0;
  //  fB_ = (kfPtr2->fx + kfPtr2->fy)/2.0;
}

void ImageMatchingAlgorithm::doSetup() {
  // Reset the match counter
  num_matches_ = 0;

  // Prepare the bookkeeping for frame A
  const size_t num_A = keyframe_A_->getNumKeypoints();
  skip_A_.clear();
  skip_A_.resize(num_A, false);
  //  Eigen::Vector3d dummy;
  //  for (size_t i = 0; i < num_A; ++i) {
  //    if (!keyframe_A_->getLandmark(i, dummy)) {
  //      skip_A_[i] = true;
  //    }
  //  }

  // Prepare the bookkeeping for frame B
  const size_t num_B = keyframe_B_->getNumKeypoints();
  skip_B_.clear();
  skip_B_.resize(num_B, false);
  //  for (size_t i = 0; i < num_B; ++i) {<
  //    if (!keyframe_B_->getLandmark(i, dummy)) {
  //      skip_B_[i] = true;
  //    }
  //  }
}

size_t ImageMatchingAlgorithm::sizeA() const {
  return keyframe_A_->getNumKeypoints();
}

size_t ImageMatchingAlgorithm::sizeB() const {
  return keyframe_B_->getNumKeypoints();
}

float ImageMatchingAlgorithm::distanceThreshold() const {
  return distance_threshold_;
}

bool ImageMatchingAlgorithm::verifyMatch(size_t indexA, size_t indexB) const {
  // Empty dummy function that just returns true;
  return true;
}

void ImageMatchingAlgorithm::reserveMatches(size_t /*numMatches*/) {
  //_triangulatedPoints.clear();
}

void ImageMatchingAlgorithm::setBestMatch(size_t index_A, size_t index_B,
                                          double distance) {
  // Create match object and push it to the matches
  Match match(index_A, index_B, distance);  // TODO: implement this
  matches_.push_back(match);

  // Increase matcher count
  num_matches_++;
}

size_t ImageMatchingAlgorithm::numMatches() { return num_matches_; }

}  // namespace pgbe
