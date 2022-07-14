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

#include <bitset>

#include <agast/glog.h>
#include <brisk/brisk.h>
#include <brisk/opencv-ref.h>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#ifndef TEST
#define TEST(a, b) void Test_##a##_##b()
#endif

TEST(Brisk, MatchBitset) {
  cv::Mat img1, img2;
  std::string image1path = "./test_data/img1.pgm";
  std::string image2path = "./test_data/img2.pgm";
  img1 = cv::imread(image1path, cv::IMREAD_GRAYSCALE);
  img2 = cv::imread(image2path, cv::IMREAD_GRAYSCALE);

  const unsigned int detection_threshold = 70;
  const unsigned int matching_threshold = 50;
  brisk::BriskFeatureDetector detector(detection_threshold, 2);
  std::vector<agast::KeyPoint> keypoints1, keypoints2;
  std::vector<std::bitset<384> > descriptors1, descriptors2;
  detector.detect(img1, keypoints1);
  detector.detect(img2, keypoints2);

  brisk::BriskDescriptorExtractor extractor;
  extractor.compute(img1, keypoints1, descriptors1);
  extractor.compute(img2, keypoints2, descriptors2);

  std::cout << "Got " << keypoints1.size() << " keypoints" << std::endl;
  std::cout << "Got " << keypoints2.size() << " keypoints" << std::endl;

  typedef std::pair<unsigned int, unsigned int> Match;
  std::vector<Match> matches;
  // Do everything without using opencv.
  for (size_t i = 0; i < keypoints1.size(); ++i) {
    unsigned int best_score = matching_threshold;
    int best_index = -1;
    for (size_t j = 0; j < keypoints2.size(); ++j) {
      unsigned int score = (descriptors1.at(i) ^ descriptors2.at(j)).count();
      if (score < best_score) {
        best_index = j;
        best_score = score;
      }
    }
    if (best_index != -1) {
      matches.emplace_back(i, best_index);
    }
  }

  Eigen::Matrix<double, 3, 3> H_1to2;
  H_1to2 << 0.8835462624646065, 0.31399802853807735, -40.079602102472926,
      -0.18170359412701342, 0.9417589525236417, 152.6910745330205,
      2.0127825613685174e-4, -1.5103648761897873e-5, 1.0;

  unsigned int outliers = 0;
  double outlier_thres = 5;

  for (const Match& match : matches) {
    Eigen::Matrix<double, 3, 1> norm1, norm2;
    norm1 << agast::KeyPointX(keypoints1.at(match.first)),
        agast::KeyPointY(keypoints1.at(match.first)), 1.0;
    norm2 << agast::KeyPointX(keypoints2.at(match.second)),
        agast::KeyPointY(keypoints2.at(match.second)), 1.0;

    Eigen::Matrix<double, 3, 1> norm1_hat = H_1to2 * norm1;
    norm1_hat /= norm1_hat(2);
    double error = (norm1_hat - norm2).norm();

    if (error > outlier_thres) {
      ++outliers;
    }
  }

  EXPECT_EQ(outliers, 0u) << "Matching outliers found (" <<
      outliers << "/" << matches.size() << ")";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
