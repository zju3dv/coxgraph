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

#ifndef BRISK_SCALE_SPACE_FEATURE_DETECTOR_H_
#define BRISK_SCALE_SPACE_FEATURE_DETECTOR_H_

#include <algorithm>
#include <limits>
#include <vector>

#include <agast/wrap-opencv.h>
#include <brisk/internal/macros.h>
#include <brisk/internal/scale-space-layer.h>

#if HAVE_OPENCV
#include <agast/glog.h>
#else
#include <glog/logging.h>
#endif

namespace brisk {

// Uses the common feature interface to construct a generic
// scale space detector from a given ScoreCalculator.
template<class SCORE_CALCULATOR_T>
#if HAVE_OPENCV
class ScaleSpaceFeatureDetector : public cv::Feature2D {
#else
class ScaleSpaceFeatureDetector {
#endif  // HAVE_OPENCV
 public:
  ScaleSpaceFeatureDetector(
      size_t octaves, double uniformityRadius, double absoluteThreshold = 0,
      size_t maxNumKpt = std::numeric_limits < size_t > ::max())
      : _octaves(octaves),
        _uniformityRadius(uniformityRadius),
        _absoluteThreshold(absoluteThreshold),
        _maxNumKpt(maxNumKpt) {
    scaleSpaceLayers.resize(std::max(_octaves * 2, size_t(1)));
  }

  typedef SCORE_CALCULATOR_T ScoreCalculator_t;
  void detect(const agast::Mat& image, std::vector<agast::KeyPoint>& keypoints,
              const agast::Mat& mask = agast::Mat()) const {
    if (image.empty()) {
      return;
    }
    CHECK(
        mask.empty()
            || (mask.type() == CV_8UC1 && mask.rows == image.rows
                && mask.cols == image.cols));
    detectImpl(image, keypoints, mask);
  }

  virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::OutputArray /*descriptors*/,
                                bool /*useProvidedKeypoints*/ = false) {
    detect(image.getMat(), keypoints, mask.getMat());
  }

 protected:
  virtual void detectImpl(const agast::Mat& image,
                          std::vector<agast::KeyPoint>& keypoints,
                          const agast::Mat& /*mask*/ = agast::Mat()) const {
    // Find out, if we should use the provided keypoints.
    bool usePassedKeypoints = false;
    if (keypoints.size() > 0)
      usePassedKeypoints = true;
    else
      keypoints.reserve(4000);  // Possibly speeds up things.

    // Construct scale space layers.
    scaleSpaceLayers[0].Create(image, !usePassedKeypoints);
    scaleSpaceLayers[0].SetUniformityRadius(_uniformityRadius);
    scaleSpaceLayers[0].SetMaxNumKpt(_maxNumKpt);
    scaleSpaceLayers[0].SetAbsoluteThreshold(_absoluteThreshold);
    for (size_t i = 1; i < _octaves * 2; ++i) {
      scaleSpaceLayers[i].Create(&scaleSpaceLayers[i - 1], !usePassedKeypoints);
      scaleSpaceLayers[i].SetUniformityRadius(_uniformityRadius);
      scaleSpaceLayers[i].SetMaxNumKpt(_maxNumKpt);
      scaleSpaceLayers[i].SetAbsoluteThreshold(_absoluteThreshold);
    }
    bool enforceUniformity = _uniformityRadius > 0.0;
    for (size_t i = 0; i < scaleSpaceLayers.size(); ++i) {
      // Only do refinement, if no keypoints are passed.
      scaleSpaceLayers[i].DetectScaleSpaceMaxima(keypoints, enforceUniformity,
                                                 !usePassedKeypoints,
                                                 usePassedKeypoints);
    }
  }

  size_t _octaves;
  double _uniformityRadius;
  double _absoluteThreshold;
  size_t _maxNumKpt;
  mutable std::vector<brisk::ScaleSpaceLayer<ScoreCalculator_t> >
    scaleSpaceLayers;
};
}  // namespace brisk

#endif  // BRISK_SCALE_SPACE_FEATURE_DETECTOR_H_
