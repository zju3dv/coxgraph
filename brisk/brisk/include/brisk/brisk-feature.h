/*
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

#ifndef BRISK_BRISK_FEATURE_H_
#define BRISK_BRISK_FEATURE_H_

#include <limits>
#include <vector>

#include <brisk/brisk-descriptor-extractor.h>
#include <brisk/brisk-feature-detector.h>
#include <agast/wrap-opencv.h>
#include <brisk/harris-feature-detector.h>
#include <brisk/harris-score-calculator.h>
#include <brisk/scale-space-feature-detector.h>

#if HAVE_OPENCV
#ifndef __ARM_NEON
namespace brisk {
class BriskFeature : public cv::Feature2D {
 public:
  BriskFeature(size_t octaves, double uniformityRadius,
               double absoluteThreshold = 0,
               size_t maxNumKpt = std::numeric_limits < size_t > ::max(),
               bool rotationInvariant = true, bool scaleInvariant = true,
               int extractorVersion=BriskDescriptorExtractor::Version::briskV2)
      : _briskDetector(octaves, uniformityRadius, absoluteThreshold, maxNumKpt),
        _briskExtractor(rotationInvariant, scaleInvariant, extractorVersion) { }

  virtual ~BriskFeature() { }

  // Inherited from cv::DescriptorExtractor interface.
  virtual int descriptorSize() const {
    return _briskExtractor.descriptorSize();
  }
  virtual int descriptorType() const {
    return _briskExtractor.descriptorType();
  }

  // Inherited from cv::Feature2D interface.
  virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                                std::vector<agast::KeyPoint>& keypoints,
                                cv::OutputArray descriptors,
                                bool useProvidedKeypoints = false) {
    if (!useProvidedKeypoints) {
      keypoints.clear();
    }

    // Convert input output arrays:
    agast::Mat descriptors_;
    agast::Mat image_ = image.getMat();
    agast::Mat mask_ = mask.getMat();

    // Run the detection. Take provided keypoints.
    _briskDetector.detect(image_, keypoints, mask_);

    // Run the extraction.
    _briskExtractor.compute(image_, keypoints, descriptors_);
    descriptors.getMatRef() = descriptors_;
  }

 protected:
  // Inherited from cv::FeatureDetector interface.
  virtual void detectImpl(const agast::Mat& image,
                          std::vector<agast::KeyPoint>& keypoints,
                          const agast::Mat& mask = agast::Mat()) const {
    _briskDetector.detect(image, keypoints, mask);
  }

  // Inherited from cv::DescriptorExtractor interface.
  virtual void computeImpl(const agast::Mat& image,
                           std::vector<agast::KeyPoint>& keypoints,
                           agast::Mat& descriptors) const {

    _briskExtractor.computeImpl(image, keypoints, descriptors);
  }

  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> _briskDetector;
  brisk::BriskDescriptorExtractor _briskExtractor;
};
}  // namespace brisk
#endif  // __ARM_NEON
#endif  // HAVE_OPENCV
#endif  // BRISK_BRISK_FEATURE_H_
