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
#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#include <emmintrin.h>
#include <tmmintrin.h>
#endif  // __ARM_NEON
#include <stdint.h>

#include <brisk/internal/vectorized-filters.h>

namespace brisk {
#ifdef __ARM_NEON
  // Not implemented.
#else
void FilterGauss3by316S(agast::Mat& src, agast::Mat& dst) {  // NOLINT
  // Sanity check.
  const unsigned int X = 3;
  const unsigned int Y = 3;
  assert(X % 2 != 0);
  assert(Y % 2 != 0);
  int cx = X / 2;
  int cy = Y / 2;

  // Dest will be 16 bit.
  dst = agast::Mat::zeros(src.rows, src.cols, CV_16S);
  const unsigned int maxJ = ((src.cols - 2) / 8) * 8;
  const unsigned int maxI = src.rows - 2;
  const unsigned int stride = src.cols;

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      // Enter convolution with kernel. do the multiplication with 2/4 at the
      // same time.
      __m128i i00 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i, j)));
      __m128i i10 = _mm_slli_epi16(
          _mm_loadu_si128(
              reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 1, j))), 1);
      __m128i i20 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 2, j)));
      __m128i i01 = _mm_slli_epi16(
          _mm_loadu_si128(
              reinterpret_cast<__m128i*>(&src.at<int16_t>(i, j + 1))), 1);
      __m128i i11 = _mm_slli_epi16(
          _mm_loadu_si128(
              reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 1, j + 1))), 2);
      __m128i i21 = _mm_slli_epi16(
          _mm_loadu_si128(
              reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 2, j + 1))), 1);
      __m128i i02 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i, j + 2)));
      __m128i i12 = _mm_slli_epi16(
          _mm_loadu_si128(
              reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 1, j + 2))), 1);
      __m128i i22 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 2, j + 2)));
      __m128i result = i11;
      // Add up.
      result = _mm_add_epi16(result, i00);
      result = _mm_add_epi16(result, i20);
      result = _mm_add_epi16(result, i02);
      result = _mm_add_epi16(result, i22);

      result = _mm_add_epi16(result, i10);
      result = _mm_add_epi16(result, i01);
      result = _mm_add_epi16(result, i12);
      result = _mm_add_epi16(result, i21);

      // Store.
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(&dst.at<int16_t>(i + cy, j + cx)), result);

      // Take care of the end.
      j += 8;
      if (j >= maxJ && !end) {
        j = stride - 2 - 8;
        end = true;
      }
    }
  }
}
#endif  // __ARM_NEON

#ifdef __ARM_NEON
  // Not implemented.
#else
void FilterGauss3by332F(agast::Mat& src, agast::Mat& dst) {  // NOLINT
  // Sanity check.
  static const unsigned int X = 3;
  static const unsigned int Y = 3;
  static const int cx = X / 2;
  static const int cy = Y / 2;

  // Destination will be 16 bit.
  dst = agast::Mat::zeros(src.rows, src.cols, CV_32F);
  const unsigned int maxJ = ((src.cols - 2) / 8) * 8;
  const unsigned int maxI = src.rows - 2;
  const unsigned int stride = src.cols;

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      // Enter convolution with kernel. do the multiplication with 2/4 at the
      // same time.
      __m128 i00 = _mm_loadu_ps(&src.at<float>(i, j));
      __m128 i10 = _mm_loadu_ps(&src.at<float>(i + 1, j));
      __m128 i20 = _mm_loadu_ps(&src.at<float>(i + 2, j));
      __m128 i01 = _mm_loadu_ps(&src.at<float>(i, j + 1));
      __m128 result = _mm_loadu_ps(&src.at<float>(i + 1, j + 1));
      __m128 i21 = _mm_loadu_ps(&src.at<float>(i + 2, j + 1));
      __m128 i02 = _mm_loadu_ps(&src.at<float>(i, j + 2));
      __m128 i12 = _mm_loadu_ps(&src.at<float>(i + 1, j + 2));
      __m128 i22 = _mm_loadu_ps(&src.at<float>(i + 2, j + 2));

      // Add up.
      result = _mm_add_ps(_mm_mul_ps(result, _mm_setr_ps(4, 4, 4, 4)), i00);
      result = _mm_add_ps(result, i20);
      result = _mm_add_ps(result, i02);
      result = _mm_add_ps(result, i22);

      __m128 result2 = _mm_add_ps(i01, i10);
      result2 = _mm_add_ps(result2, i12);
      result2 = _mm_add_ps(result2, i21);
      result = _mm_add_ps(_mm_mul_ps(result2, _mm_setr_ps(2, 2, 2, 2)), result);
      result = _mm_mul_ps(result2, _mm_setr_ps(0.0625, 0.0625, 0.0625, 0.0625));

      // Store.
      _mm_storeu_ps(&dst.at<float>(i + cy, j + cx), result);

      // Take care about end.
      j += 4;
      if (j >= maxJ && !end) {
        j = stride - 2 - 4;
        end = true;
      }
    }
  }
}
#endif  // __ARM_NEON

#ifdef __ARM_NEON
// Not implemented.
#else
void FilterBox3by316S(agast::Mat& src, agast::Mat& dst) {  // NOLINT
  // Sanity check.
  const unsigned int X = 3;
  const unsigned int Y = 3;
  assert(X % 2 != 0);
  assert(Y % 2 != 0);
  int cx = X / 2;
  int cy = Y / 2;

  // Destination will be 16 bit.
  dst = agast::Mat::zeros(src.rows, src.cols, CV_16S);
  const unsigned int maxJ = ((src.cols - 2) / 8) * 8;
  const unsigned int maxI = src.rows - 2;
  const unsigned int stride = src.cols;

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      // Enter convolution with kernel. do the multiplication with 2/4 at the
      // same time.
      __m128i i00 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i, j)));
      __m128i i10 = (_mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 1, j))));
      __m128i i20 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 2, j)));
      __m128i i01 = (_mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i, j + 1))));
      __m128i i11 = (_mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 1, j + 1))));
      __m128i i21 = (_mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 2, j + 1))));
      __m128i i02 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i, j + 2)));
      __m128i i12 = (_mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 1, j + 2))));
      __m128i i22 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&src.at<int16_t>(i + 2, j + 2)));
      __m128i result = i11;
      // Add up.
      result = _mm_add_epi16(result, i00);
      result = _mm_add_epi16(result, i20);
      result = _mm_add_epi16(result, i02);
      result = _mm_add_epi16(result, i22);

      result = _mm_add_epi16(result, i10);
      result = _mm_add_epi16(result, i01);
      result = _mm_add_epi16(result, i12);
      result = _mm_add_epi16(result, i21);

      // Store.
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(&dst.at<int16_t>(i + cy, j + cx)), result);

      // Take care about end.
      j += 8;
      if (j >= maxJ && !end) {
        j = stride - 2 - 8;
        end = true;
      }
    }
  }
}
#endif  // __ARM_NEON
}  // namespace brisk
