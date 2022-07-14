/*
 Copyright (C) 2011  The Autonomous Systems Lab, ETH Zurich,
 Stefan Leutenegger, Simon Lynen and Margarita Chli.

 Copyright (C) 2013  The Autonomous Systems Lab, ETH Zurich,
 Stefan Leutenegger and Simon Lynen.

 BRISK - Binary Robust Invariant Scalable Keypoints
 Reference implementation of
 [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
 Binary Robust Invariant Scalable Keypoints, in Proceedings of
 the IEEE International Conference on Computer Vision (ICCV2011).

 This file is part of BRISK.

 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of the <organization> nor the
       names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>

#include <agast/glog.h>
#include <agast/wrap-opencv.h>
#include <brisk/brisk-feature-detector.h>
#include <brisk/internal/brisk-scale-space.h>

namespace {
void RemoveInvalidKeyPoints(const agast::Mat& mask,
                            std::vector<agast::KeyPoint>* keypoints) {
  CHECK_NOTNULL(keypoints);
  if (mask.empty())
    return;

  std::function<bool(const agast::KeyPoint& key_pt)> masking =
      [&mask](const agast::KeyPoint& key_pt)->bool {
        const float& keypoint_x = agast::KeyPointX(key_pt);
        const float& keypoint_y = agast::KeyPointY(key_pt);
        return mask.at<unsigned char>(static_cast<int>(keypoint_y + 0.5f),
            static_cast<int>(keypoint_x + 0.5f)) == 0;
  };

  keypoints->erase(
      std::remove_if(keypoints->begin(), keypoints->end(),
                     masking), keypoints->end());
}
}  // namespace

namespace brisk {
BriskFeatureDetector::BriskFeatureDetector(int thresh, int octaves,
                                           bool suppressScaleNonmaxima) {
  threshold = thresh;
  this->octaves = octaves;
  m_suppressScaleNonmaxima = suppressScaleNonmaxima;
}

void BriskFeatureDetector::detectImpl(const agast::Mat& image,
                                      std::vector<agast::KeyPoint>& keypoints,
                                      const agast::Mat& mask) const {
  keypoints.clear();
  brisk::BriskScaleSpace briskScaleSpace(octaves, m_suppressScaleNonmaxima);
  briskScaleSpace.ConstructPyramid(image, threshold);
  briskScaleSpace.GetKeypoints(&keypoints);
  RemoveInvalidKeyPoints(mask, &keypoints);
}

void BriskFeatureDetector::ComputeScale(
    const agast::Mat& image, std::vector<agast::KeyPoint>& keypoints) const {
  BriskScaleSpace briskScaleSpace(octaves, m_suppressScaleNonmaxima);
  briskScaleSpace.ConstructPyramid(image, threshold, 0);
  briskScaleSpace.GetKeypoints(&keypoints);
}
}  // namespace brisk
