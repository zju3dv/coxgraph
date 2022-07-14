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
 * frame-noncentral-absolute-adapter.hpp
 * @brief Source file for the FrameNoncentralAbsoluteAdapter Class
 * @author: Marco Karrer
 * Created on: Aug 16, 2018
 */

#pragma once

#include <stdlib.h>
#include <memory>
#include <vector>

#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/types.hpp>

#include "pose_graph_backend/keyframe.hpp"
#include "typedefs.hpp"

namespace opengv {

namespace absolute_pose {

using namespace pgbe;

/// @brief Adapter for absolute pose RANSAC (3D2D) with non-central cameras,
///        i.e. could be a multi-camera-setup.
class FrameNoncentralAbsoluteAdapter : public AbsoluteAdapterBase {
 private:
  using AbsoluteAdapterBase::_R;
  using AbsoluteAdapterBase::_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// @brief type for describing matches.
  typedef std::vector<int> matches_t;

  /// @brief Constructor.
  /// @param keyframePtr   The keyframe.
  /// @param matchedPoints The matched 3d points
  FrameNoncentralAbsoluteAdapter(const std::shared_ptr<KeyFrame> keyframe_ptr,
                                 const std::shared_ptr<KeyFrame> landmark_kf,
                                 const pgbe::Matches& matches) {
    // Only a single camera without offset
    camOffsets_.push_back(Eigen::Vector3d::Zero());
    camRotations_.push_back(Eigen::Matrix3d::Identity());
    const double fu = keyframe_ptr->getFocalLength();

    // Extract the geometrical information
    int noCorrespondences = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
      Eigen::Vector3d landmark;

      if (!landmark_kf->getLandmark(matches[i].idx_A, landmark)) {
        continue;
      }

      // Add the landmark
      points_.push_back(landmark);

      // Add the bearing vector and the sigma of the angle
      double keypointStdDev = 2.0;
      bearingVectors_.push_back(
          keyframe_ptr->getKeypointBearing(matches[i].idx_B));
      sigmaAngles_.push_back(sqrt(2) * keypointStdDev * keypointStdDev /
                             (fu * fu));

      // count
      ++noCorrespondences;

      // store camera index (only a single camera --> 0)
      camIndices_.push_back(0);

      // store keypoint index
      keypointIndices_.push_back(matches[i].idx_A);
    }
  }

  virtual ~FrameNoncentralAbsoluteAdapter() {}

  /// @brief Retrieve the bearing vector of a correspondence.
  /// @param index The serialized index of the correspondence.
  /// @return The corresponding bearing vector.
  virtual opengv::bearingVector_t getBearingVector(size_t index) const {
    assert(index < bearingVectors_.size());
    return bearingVectors_[index];
  }

  /// @brief Retrieve the position of a camera of a correspondence
  ///        seen from the viewpoint origin.
  /// @param index The serialized index of the correspondence.
  /// @return The position of the corresponding camera seen from the viewpoint
  /// origin.
  virtual opengv::translation_t getCamOffset(size_t index) const {
    return camOffsets_[camIndices_[index]];
  }

  /// @brief Retrieve the rotation from a camera of a correspondence to the
  ///        viewpoint origin.
  /// @param index The serialized index of the correspondence.
  ///  @return The rotation from the corresponding camera back to the viewpoint
  ///       origin.
  virtual opengv::rotation_t getCamRotation(size_t index) const {
    return camRotations_[camIndices_[index]];
  }

  /// @brief Retrieve the world point of a correspondence.
  /// @param index The serialized index of the correspondence.
  /// @return The corresponding world point.
  virtual opengv::point_t getPoint(size_t index) const {
    assert(index < bearingVectors_.size());
    return points_[index];
  }

  /// @brief Get the number of correspondences. These are keypoints that have a
  ///        corresponding landmark which is added to the estimator,
  ///        has more than one observation and not at infinity.
  /// @return Number of correspondences.
  virtual size_t getNumberCorrespondences() const { return points_.size(); }

  //// @brief Get the camera index for a specific correspondence.
  /// @param index The serialized index of the correspondence.
  /// @return Camera index of the correspondence.
  int camIndex(size_t index) const { return camIndices_.at(index); }

  /// @brief Get the keypoint index for a specific correspondence
  /// @param index The serialized index of the correspondence.
  /// @return Keypoint index belonging to the correspondence.
  int keypointIndex(size_t index) const { return keypointIndices_.at(index); }

  /// @brief Retrieve the weight of a correspondence. The weight is supposed to
  ///       reflect the quality of a correspondence, and typically is between
  ///        0 and 1.
  /// @warning This is not implemented and always returns 1.0.
  virtual double getWeight(size_t) const { return 1.0; }

  /// @brief Obtain the angular standard deviation in [rad].
  /// @param index The index of the correspondence.
  /// @return The standard deviation in [rad].
  double getSigmaAngle(size_t index) { return sigmaAngles_[index]; }

 private:
  /// The bearing vectors of the correspondences.
  opengv::bearingVectors_t bearingVectors_;

  /// The world coordinates of the correspondences.
  opengv::points_t points_;

  /// The camera indices of the correspondences.
  std::vector<size_t> camIndices_;

  /// The keypoint indices of the correspondences.
  std::vector<size_t> keypointIndices_;

  /// The position of the cameras seen from the viewpoint origin
  opengv::translations_t camOffsets_;

  /// The rotation of the cameras to the viewpoint origin.
  opengv::rotations_t camRotations_;

  /// The standard deviations of the bearing vectors in [rad].
  std::vector<double> sigmaAngles_;
};

}  // namespace absolute_pose
}  // namespace opengv
