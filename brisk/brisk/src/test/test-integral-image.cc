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

#include <agast/glog.h>
#include <agast/wrap-opencv.h>
#include <gtest/gtest.h>

#include <brisk/internal/integral-image.h>

#ifndef TEST
#define TEST(a, b) int Test_##a##_##b()
#endif

void ComputeReferenceIntegralImage8Bit(const agast::Mat& src, agast::Mat* dest) {
  CHECK_NOTNULL(dest);
  int x, y;
  const int cn = 1;
  unsigned char var;
  const int srcstep = static_cast<int>(src.step / sizeof(var));

  dest->create(src.rows + 1, src.cols + 1, CV_32SC1);

  uint32_t var2;
  const int sumstep = static_cast<int>(dest->step / sizeof(var2));

  uint32_t* sum = (uint32_t*) (dest->data);
  unsigned char* _src = static_cast<unsigned char*>(src.data);

  memset(sum, 0, (src.cols + cn) * sizeof(sum[0]));
  sum += sumstep + 1;

  for (y = 0; y < src.rows; y++, _src += srcstep, sum += sumstep) {
    uint32_t s = sum[-1] = 0;
    for (x = 0; x < src.cols; ++x) {
      s += _src[x];
      sum[x] = sum[x - sumstep] + s;
    }
  }
}

TEST(Brisk, IntegralImage8bit) {
  std::string imagepath = "./test_data/img1.pgm";
  cv::Mat src_img = cv::imread(imagepath, cv::IMREAD_GRAYSCALE);

  agast::Mat integral, integral_verification;
  ComputeReferenceIntegralImage8Bit(src_img, &integral_verification);
  brisk::IntegralImage8(src_img, &integral);

  ASSERT_EQ(integral.rows, integral_verification.rows);
  ASSERT_EQ(integral.cols, integral_verification.cols);

  int errors = 0;
  const int kMaxErrors = 10;
  for (int row = 0; row < integral.rows; ++row) {
    for (int col = 0; col < integral.cols; ++col) {
      EXPECT_EQ(integral.at<int>(row, col),
                integral_verification.at<int>(row, col));
      if (integral.at<int>(row, col)
          != integral_verification.at<int>(row, col)) {
        ++errors;
        CHECK_LT(errors, kMaxErrors);
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
