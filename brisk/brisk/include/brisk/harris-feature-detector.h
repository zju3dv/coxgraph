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

#ifndef BRISK_HARRIS_FEATURE_DETECTOR_H_
#define BRISK_HARRIS_FEATURE_DETECTOR_H_

#include <vector>

#include <agast/wrap-opencv.h>
#include <brisk/internal/macros.h>
#include <brisk/internal/vectorized-filters.h>
#ifdef __ARM_NEON
// Not implemented.
#else
namespace brisk {
#if HAVE_OPENCV
class HarrisFeatureDetector : public cv::FeatureDetector {
#else
class HarrisFeatureDetector {
#endif  // HAVE_OPENCV
 public:
  explicit HarrisFeatureDetector(double radius);
  void SetRadius(double radius);

 protected:
  static __inline__ void GetCovarEntries(const agast::Mat& src, agast::Mat& dxdx,
                                         agast::Mat& dydy, agast::Mat& dxdy);
  static __inline__ void CornerHarris(const agast::Mat& dxdxSmooth,
                                      const agast::Mat& dydySmooth,
                                      const agast::Mat& dxdySmooth,
                                      agast::Mat& score);
  static __inline__ void NonmaxSuppress(const agast::Mat& scores,
                                        std::vector<agast::KeyPoint>& keypoints);
  __inline__ void EnforceUniformity(const agast::Mat& scores,
                                    std::vector<agast::KeyPoint>& keypoints) const;

  virtual void detectImpl(const agast::Mat& image,
                          std::vector<agast::KeyPoint>& keypoints,
                          const agast::Mat& mask = agast::Mat()) const;

  double _radius;
  agast::Mat _LUT;
};
}  // namespace brisk
#endif  // __ARM_NEON
#endif  // BRISK_HARRIS_FEATURE_DETECTOR_H_
