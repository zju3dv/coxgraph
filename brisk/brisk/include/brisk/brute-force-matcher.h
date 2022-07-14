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

#ifndef BRISK_BRUTE_FORCE_MATCHER_H_
#define BRISK_BRUTE_FORCE_MATCHER_H_

#include <vector>

#include <agast/wrap-opencv.h>
#include <brisk/internal/hamming.h>
#include <brisk/internal/macros.h>


namespace brisk {
class BruteForceMatcher;
#if HAVE_OPENCV
class  BruteForceMatcher : public cv::DescriptorMatcher {
 public:
  BruteForceMatcher(const brisk::Hamming& distance = brisk::Hamming())
      : distance_(distance) { }
  virtual ~BruteForceMatcher() { }
  virtual bool isMaskSupported() const {
    return true;
  }
  virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false)
      const;

 protected:
  virtual void knnMatchImpl(
      cv::InputArray queryDescriptors,
      std::vector<std::vector<cv::DMatch>>& matches, int k,
      cv::InputArrayOfArrays masks = cv::noArray(),
      bool compactResult = false);
  void radiusMatchImpl(
      cv::InputArray& queryDescriptors,
      std::vector<std::vector<cv::DMatch> >& matches, float maxDistance,
      cv::InputArrayOfArrays masks = cv::noArray(),
      bool compactResult = false);

  brisk::Hamming distance_;

 private:
  //  Next two methods are used to implement specialization.
  static void commonKnnMatchImpl(BruteForceMatcher& matcher,  // NOLINT
                                 const agast::Mat& queryDescriptors,
                                 std::vector<std::vector<cv::DMatch> >& matches,
                                 int k,
                                 const std::vector<agast::Mat>& masks,
                                 bool compactResult);
  static void commonRadiusMatchImpl(
      BruteForceMatcher& matcher,  // NOLINT
      const agast::Mat& queryDescriptors,
      std::vector<std::vector<cv::DMatch> >& matches,
      float maxDistance,
      const std::vector<agast::Mat>& masks,
      bool compactResult);
};
#endif  // HAVE_OPENCV
}  // namespace brisk
#endif  // BRISK_BRUTE_FORCE_MATCHER_H_
