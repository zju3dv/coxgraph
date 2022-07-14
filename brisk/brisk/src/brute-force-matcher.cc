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

#include <memory>

#include <brisk/brute-force-matcher.h>
#include <agast/wrap-opencv.h>

#if HAVE_OPENCV
namespace brisk {
// Adapted from OpenCV 2.3 features2d/matcher.hpp
cv::Ptr<cv::DescriptorMatcher> BruteForceMatcher::clone(bool emptyTrainData)
const {
  BruteForceMatcher* matcher = new BruteForceMatcher(distance_);
  if (!emptyTrainData) {
    std::transform(trainDescCollection.begin(), trainDescCollection.end(),
                   matcher->trainDescCollection.begin(), clone_op);
  }
  return matcher;
}

void BruteForceMatcher::knnMatchImpl(
    cv::InputArray queryDescriptors,
    std::vector<std::vector<cv::DMatch>>& matches, int k,
    cv::InputArrayOfArrays masks,
    bool compactResult) {
  std::vector<agast::Mat> masks_mat_vector;
  masks.getMatVector(masks_mat_vector);
  commonKnnMatchImpl(*this, queryDescriptors.getMat(), matches, k,
                     masks_mat_vector, compactResult);
}

void BruteForceMatcher::radiusMatchImpl(
    cv::InputArray& queryDescriptors,
    std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
    cv::InputArrayOfArrays masks, bool compactResult) {
  std::vector<agast::Mat> masks_mat_vector;
  masks.getMatVector(masks_mat_vector);
  commonRadiusMatchImpl(*this, queryDescriptors.getMat(), matches, maxDistance,
                        masks_mat_vector, compactResult);
}

inline void BruteForceMatcher::commonKnnMatchImpl(
    BruteForceMatcher& matcher, const agast::Mat& queryDescriptors,
    std::vector<std::vector<cv::DMatch> >& matches,
    int knn,
    const std::vector<agast::Mat>& masks,
    bool compactResult) {
  typedef brisk::Hamming::ValueType ValueType;
  typedef brisk::Hamming::ResultType DistanceType;
  assert(!queryDescriptors.empty());
  assert(cv::DataType<ValueType>::type == queryDescriptors.type());

  int dimension = queryDescriptors.cols;
  matches.reserve(queryDescriptors.rows);

  size_t imgCount = matcher.trainDescCollection.size();
  // Distances between one query descriptor and all train descriptors.
  std::vector<agast::Mat> allDists(imgCount);
  for (size_t i = 0; i < imgCount; i++)
    allDists[i] = agast::Mat(1, matcher.trainDescCollection[i].rows,
                          cv::DataType<DistanceType>::type);

  for (int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++) {
    if (matcher.isMaskedOut(masks, qIdx)) {
      if (!compactResult)  // Push empty vector.
        matches.push_back(std::vector<cv::DMatch>());
    } else {
      // 1. compute distances between i-th query descriptor and all train
      // descriptors.
      for (size_t iIdx = 0; iIdx < imgCount; iIdx++) {
        assert(
            cv::DataType<ValueType>::type
                == matcher.trainDescCollection[iIdx].type()
                || matcher.trainDescCollection[iIdx].empty());
        assert(
            queryDescriptors.cols == matcher.trainDescCollection[iIdx].cols
                || matcher.trainDescCollection[iIdx].empty());

        const ValueType* d1 = (const ValueType*) (queryDescriptors.data
            + queryDescriptors.step * qIdx);
        allDists[iIdx].setTo(
            cv::Scalar::all(std::numeric_limits<DistanceType>::max()));
        for (int tIdx = 0; tIdx < matcher.trainDescCollection[iIdx].rows;
            tIdx++) {
          if (masks.empty()
              || matcher.isPossibleMatch(masks[iIdx], qIdx, tIdx)) {
            const ValueType* d2 = (const ValueType*) (matcher
                .trainDescCollection[iIdx].data
                + matcher.trainDescCollection[iIdx].step * tIdx);
            allDists[iIdx].at<DistanceType>(0, tIdx) =
                matcher.distance_(d1, d2, dimension);
          }
        }
      }

      // 2. choose k nearest matches for query[i].
      matches.push_back(std::vector<cv::DMatch>());
      std::vector<std::vector<cv::DMatch> >::reverse_iterator curMatches =
          matches.rbegin();
      for (int k = 0; k < knn; k++) {
        cv::DMatch bestMatch;
        bestMatch.distance = std::numeric_limits<float>::max();
        for (size_t iIdx = 0; iIdx < imgCount; iIdx++) {
          if (!allDists[iIdx].empty()) {
            double minVal;
            cv::Point minLoc;
            minMaxLoc(allDists[iIdx], &minVal, 0, &minLoc, 0);
            if (minVal < bestMatch.distance)
              bestMatch = cv::DMatch(qIdx, minLoc.x, static_cast<int>(iIdx),
                                     static_cast<float>(minVal));
          }
        }
        if (bestMatch.trainIdx == -1)
          break;

        allDists[bestMatch.imgIdx].at<DistanceType> (0, bestMatch.trainIdx) =
            std::numeric_limits < DistanceType > ::max();
        curMatches->push_back(bestMatch);
      }
      // TODO(slynen): Shouldn't this be already sorted at this point?
      std::sort(curMatches->begin(), curMatches->end());
    }
  }
}

inline void BruteForceMatcher::commonRadiusMatchImpl(
    BruteForceMatcher& matcher, const agast::Mat& queryDescriptors,
    std::vector<std::vector<cv::DMatch> >& matches, float maxDistance,
    const std::vector<agast::Mat>& masks, bool compactResult) {
  typedef brisk::Hamming::ValueType ValueType;
  typedef brisk::Hamming::ResultType DistanceType;
  CV_DbgAssert(!queryDescriptors.empty());
  assert(cv::DataType < ValueType > ::type == queryDescriptors.type());
  int dimension = queryDescriptors.cols;

  matches.reserve(queryDescriptors.rows);

  size_t imgCount = matcher.trainDescCollection.size();
  for (int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++) {
    if (matcher.isMaskedOut(masks, qIdx)) {
      if (!compactResult)  // Push empty vector.
        matches.push_back(std::vector<cv::DMatch>());
    } else {
      matches.push_back(std::vector<cv::DMatch>());
      std::vector<std::vector<cv::DMatch> >::reverse_iterator curMatches =
          matches.rbegin();
      for (size_t iIdx = 0; iIdx < imgCount; iIdx++) {
        assert(
            cv::DataType < ValueType > ::type
                == matcher.trainDescCollection[iIdx].type()
                || matcher.trainDescCollection[iIdx].empty());
        assert(
            queryDescriptors.cols == matcher.trainDescCollection[iIdx].cols
                || matcher.trainDescCollection[iIdx].empty());

        const ValueType* d1 = (const ValueType*) (queryDescriptors.data +
            queryDescriptors.step * qIdx);
        for (int tIdx = 0; tIdx < matcher.trainDescCollection[iIdx].rows;
            tIdx++) {
          if (masks.empty()
              || matcher.isPossibleMatch(masks[iIdx], qIdx, tIdx)) {
            const ValueType* d2 = static_cast<const ValueType*>(
                matcher.trainDescCollection[iIdx].data
                + matcher.trainDescCollection[iIdx].step * tIdx);
            DistanceType d = matcher.distance_(d1, d2, dimension);
            if (d < maxDistance)
              curMatches->push_back(cv::DMatch(qIdx, tIdx,
                                               static_cast<int>(iIdx),
                                               static_cast<float>(d)));
          }
        }
      }
      std::sort(curMatches->begin(), curMatches->end());
    }
  }
}
}  // namespace brisk
#endif  // HAVE_OPENCV
