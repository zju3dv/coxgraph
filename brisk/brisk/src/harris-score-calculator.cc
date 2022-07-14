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
// Not implemented.
#else
#include <emmintrin.h>
#include <tmmintrin.h>
#include <stdint.h>

#include <brisk/harris-score-calculator.h>
#include <brisk/internal/harris-scores.h>

namespace brisk {

void HarrisScoreCalculator::InitializeScores() {
  HarrisScoresSSE(_img, _scores);
}

void HarrisScoreCalculator::Get2dMaxima(std::vector<PointWithScore>& points,  // NOLINT
                                        int absoluteThreshold) {
  // Do the 8-neighbor nonmax suppression.
  const int stride = _scores.cols;
  const int rows_end = _scores.rows - 2;
  points.reserve(4000);
  for (int j = 2; j < rows_end; ++j) {
    const int* p = &_scores.at<int>(j, 2);
    const int* const p_begin = p;
    const int* const p_end = &_scores.at<int>(j, stride - 2);
    bool last = false;
    while (p < p_end) {
      const int center = *p;
      const int* const center_p = p;
      ++p;
      if (last) {
        last = false;
        continue;
      }
      if (center < absoluteThreshold)
        continue;
      if (*(center_p + 1) > center)
        continue;
      if (*(center_p - 1) > center)
        continue;
      const int* const p1 = (center_p + stride);
      if (*p1 > center)
        continue;
      const int* const p2 = (center_p - stride);
      if (*p2 > center)
        continue;
      if (*(p1 + 1) > center)
        continue;
      if (*(p1 - 1) > center)
        continue;
      if (*(p2 + 1) > center)
        continue;
      if (*(p2 - 1) > center)
        continue;
      const int i = center_p - p_begin + 2;

#ifdef USE_SIMPLE_POINT_WITH_SCORE
      points.push_back(PointWithScore(center, i, j));
#else
#error
      points.push_back(PointWithScore(cv::Point2i(i, j), center));
#endif
    }
  }
}

// X and Y denote the size of the mask.
void HarrisScoreCalculator::GetCovarEntries(const agast::Mat& src, agast::Mat& dxdx,
                                            agast::Mat& dydy, agast::Mat& dxdy) {
  // Sanity check.
  assert(src.type() == CV_8U);
  agast::Mat kernel = agast::Mat::zeros(3, 3, CV_16S);
  kernel.at<int16_t>(0, 0) = 3 * 8;
  kernel.at<int16_t>(1, 0) = 10 * 8;
  kernel.at<int16_t>(2, 0) = 3 * 8;
  kernel.at<int16_t>(0, 2) = -3 * 8;
  kernel.at<int16_t>(1, 2) = -10 * 8;
  kernel.at<int16_t>(2, 2) = -3 * 8;

  const unsigned int X = 3;
  const unsigned int Y = 3;
  const unsigned int cx = 1;
  const unsigned int cy = 1;

  // Dest will be 16 bit.
  dxdx = agast::Mat::zeros(src.rows, src.cols, CV_16S);
  dydy = agast::Mat::zeros(src.rows, src.cols, CV_16S);
  dxdy = agast::Mat::zeros(src.rows, src.cols, CV_16S);

  const unsigned int maxJ = ((src.cols - 2) / 16) * 16;
  const unsigned int maxI = src.rows - 2;
  const unsigned int stride = src.cols;

  __m128i mask_hi = _mm_set_epi8(0, -1, 0, -1, 0, -1, 0, -1,
                                 0, -1, 0, -1, 0, -1, 0, -1);
  __m128i mask_lo = _mm_set_epi8(-1, 0, -1, 0, -1, 0, -1, 0,
                                 -1, 0, -1, 0, -1, 0, -1, 0);

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      __m128i zeros = _mm_setzero_si128();
      __m128i result_hi_dx = zeros;
      __m128i result_lo_dx = zeros;
      __m128i result_hi_dy = zeros;
      __m128i result_lo_dy = zeros;
      // Enter convolution with kernel.
      for (unsigned int x = 0; x < X; ++x) {
        for (unsigned int y = 0; y < Y; ++y) {
          const int16_t& m_dx = kernel.at<int16_t>(y, x);
          const int16_t& m_dy = kernel.at<int16_t>(x, y);
          __m128i mult_dx = _mm_set_epi16(m_dx, m_dx, m_dx, m_dx, m_dx, m_dx,
                                          m_dx, m_dx);
          __m128i mult_dy = _mm_set_epi16(m_dy, m_dy, m_dy, m_dy, m_dy, m_dy,
                                          m_dy, m_dy);
          unsigned char* p = (src.data + (stride * (i + y)) + x + j);
          __m128i i0 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
          __m128i i0_hi = _mm_and_si128(i0, mask_hi);
          __m128i i0_lo = _mm_srli_si128(_mm_and_si128(i0, mask_lo), 1);

          if (m_dx != 0) {
            __m128i i_hi_dx = _mm_mullo_epi16(i0_hi, mult_dx);
            __m128i i_lo_dx = _mm_mullo_epi16(i0_lo, mult_dx);
            result_hi_dx = _mm_add_epi16(result_hi_dx, i_hi_dx);
            result_lo_dx = _mm_add_epi16(result_lo_dx, i_lo_dx);
          }

          if (m_dy != 0) {
            __m128i i_hi_dy = _mm_mullo_epi16(i0_hi, mult_dy);
            __m128i i_lo_dy = _mm_mullo_epi16(i0_lo, mult_dy);
            result_hi_dy = _mm_add_epi16(result_hi_dy, i_hi_dy);
            result_lo_dy = _mm_add_epi16(result_lo_dy, i_lo_dy);
          }
        }
      }

      // Calculate covariance entries - remove precision (ends up being 4 bit),
      // then remove 4 more bits.
      __m128i i_hi_dx_dx = _mm_srai_epi16(
          _mm_mulhi_epi16(result_hi_dx, result_hi_dx), 4);
      __m128i i_hi_dy_dy = _mm_srai_epi16(
          _mm_mulhi_epi16(result_hi_dy, result_hi_dy), 4);
      __m128i i_hi_dx_dy = _mm_srai_epi16(
          _mm_mulhi_epi16(result_hi_dy, result_hi_dx), 4);
      __m128i i_lo_dx_dx = _mm_srai_epi16(
          _mm_mulhi_epi16(result_lo_dx, result_lo_dx), 4);
      __m128i i_lo_dy_dy = _mm_srai_epi16(
          _mm_mulhi_epi16(result_lo_dy, result_lo_dy), 4);
      __m128i i_lo_dx_dy = _mm_srai_epi16(
          _mm_mulhi_epi16(result_lo_dy, result_lo_dx), 4);

      // Store.
      unsigned char* p_lo_dxdx = (dxdx.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j;
      unsigned char* p_hi_dxdx = (dxdx.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j
          + 16;
      _mm_storeu_si128(reinterpret_cast<__m128i *>(p_hi_dxdx),
                       _mm_unpackhi_epi16(i_hi_dx_dx, i_lo_dx_dx));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(p_lo_dxdx),
                       _mm_unpacklo_epi16(i_hi_dx_dx, i_lo_dx_dx));
      unsigned char* p_lo_dydy = (dydy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j;
      unsigned char* p_hi_dydy = (dydy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j
          + 16;
      _mm_storeu_si128(reinterpret_cast<__m128i *>(p_hi_dydy),
                       _mm_unpackhi_epi16(i_hi_dy_dy, i_lo_dy_dy));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(p_lo_dydy),
                       _mm_unpacklo_epi16(i_hi_dy_dy, i_lo_dy_dy));
      unsigned char* p_lo_dxdy = (dxdy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j;
      unsigned char* p_hi_dxdy = (dxdy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j
          + 16;
      _mm_storeu_si128(reinterpret_cast<__m128i *>(p_hi_dxdy),
                       _mm_unpackhi_epi16(i_hi_dx_dy, i_lo_dx_dy));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(p_lo_dxdy),
                       _mm_unpacklo_epi16(i_hi_dx_dy, i_lo_dx_dy));

      // Take care about end.
      j += 16;
      if (j >= maxJ && !end) {
        j = stride - 2 - 16;
        end = true;
      }
    }
  }
}

void HarrisScoreCalculator::CornerHarris(const agast::Mat& dxdxSmooth,
                                         const agast::Mat& dydySmooth,
                                         const agast::Mat& dxdySmooth,
                                         agast::Mat& dst) {
  // Dest will be 16 bit.
  dst = agast::Mat::zeros(dxdxSmooth.rows, dxdxSmooth.cols, CV_32S);
  const unsigned int maxJ = ((dxdxSmooth.cols - 2) / 8) * 8;
  const unsigned int maxI = dxdxSmooth.rows - 2;
  const unsigned int stride = dxdxSmooth.cols;

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      __m128i dxdx = _mm_loadu_si128(
          reinterpret_cast<const __m128i *>(&dxdxSmooth.at<int16_t>(i, j)));
      __m128i dydy = _mm_loadu_si128(
          reinterpret_cast<const __m128i *>(&dydySmooth.at<int16_t>(i, j)));
      __m128i dxdy = _mm_loadu_si128(
          reinterpret_cast<const __m128i *>(&dxdySmooth.at<int16_t>(i, j)));

      // Determinant terms.
      __m128i prod1_lo = _mm_mullo_epi16(dxdx, dydy);
      __m128i prod1_hi = _mm_mulhi_epi16(dxdx, dydy);
      __m128i prod2_lo = _mm_mullo_epi16(dxdy, dxdy);
      __m128i prod2_hi = _mm_mulhi_epi16(dxdy, dxdy);
      __m128i prod1_1 = _mm_unpacklo_epi16(prod1_lo, prod1_hi);
      __m128i prod1_2 = _mm_unpackhi_epi16(prod1_lo, prod1_hi);
      __m128i prod2_1 = _mm_unpacklo_epi16(prod2_lo, prod2_hi);
      __m128i prod2_2 = _mm_unpackhi_epi16(prod2_lo, prod2_hi);

      // Calculate the determinant.
      __m128i det_1 = _mm_sub_epi32(prod1_1, prod2_1);
      __m128i det_2 = _mm_sub_epi32(prod1_2, prod2_2);

      // Trace - uses kappa = 1 / 16.
      __m128i trace_quarter = _mm_srai_epi16(
          _mm_add_epi16(_mm_srai_epi16(dxdx, 1), _mm_srai_epi16(dydy, 1)), 1);
      __m128i trace_sq_00625_lo = _mm_mullo_epi16(trace_quarter, trace_quarter);
      __m128i trace_sq_00625_hi = _mm_mulhi_epi16(trace_quarter, trace_quarter);
      __m128i trace_sq_00625_1 = _mm_unpacklo_epi16(trace_sq_00625_lo,
                                                    trace_sq_00625_hi);
      __m128i trace_sq_00625_2 = _mm_unpackhi_epi16(trace_sq_00625_lo,
                                                    trace_sq_00625_hi);

      // Form score.
      __m128i score_1 = _mm_sub_epi32(det_1, trace_sq_00625_1);
      __m128i score_2 = _mm_sub_epi32(det_2, trace_sq_00625_2);

      // Store.
      _mm_storeu_si128(reinterpret_cast<__m128i *>(&dst.at<int>(i, j)),
                       score_1);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(&dst.at<int>(i, j + 4)),
                       score_2);

      // Take care about end.
      j += 8;
      if (j >= maxJ && !end) {
        j = stride - 2 - 8;
        end = true;
      }
    }
  }
}
}  // namespace brisk
#endif  // __ARM_NEON
