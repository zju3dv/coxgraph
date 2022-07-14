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
#include <tmmintrin.h>
#include <mmintrin.h>
#include <stdint.h>

#include <brisk/harris-feature-detector.h>

namespace brisk {

__inline__ bool compareKeypointScore(agast::KeyPoint kp_i, agast::KeyPoint kp_j) {
  return (agast::KeyPointResponse(kp_i) > agast::KeyPointResponse(kp_j));
}

HarrisFeatureDetector::HarrisFeatureDetector(double radius) {
  SetRadius(radius);
}
void HarrisFeatureDetector::SetRadius(double radius) {
  _radius = radius;

  // Generic mask.
  _LUT = agast::Mat::zeros(2 * 16 - 1, 2 * 16 - 1, CV_32F);
  for (int x = 0; x < 2 * 16 - 1; ++x) {
    for (int y = 0; y < 2 * 16 - 1; ++y) {
      _LUT.at<float>(y, x) = std::max(
          1 - static_cast<double>((radius / 2.0 - x) * (radius / 2.0 - x)
                      + (radius / 2.0 - y) * (radius / 2.0 - y))
                  / static_cast<double>(radius / 2.0 * radius / 2.0), 0.0);
    }
  }
}

// X and Y denote the size of the mask.
__inline__ void HarrisFeatureDetector::GetCovarEntries(const agast::Mat& src,
                                                       agast::Mat& dxdx,
                                                       agast::Mat& dydy,
                                                       agast::Mat& dxdy) {
  // Sanity check.
  agast::Mat kernel = agast::Mat::zeros(3, 3, CV_8S);
  kernel.at<char>(0, 0) = 3 * 8;
  kernel.at<char>(1, 0) = 10 * 8;
  kernel.at<char>(2, 0) = 3 * 8;
  kernel.at<char>(0, 2) = -3 * 8;
  kernel.at<char>(1, 2) = -10 * 8;
  kernel.at<char>(2, 2) = -3 * 8;

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
      __m128i result_hi_dx = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
      __m128i result_lo_dx = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
      __m128i result_hi_dy = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
      __m128i result_lo_dy = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
      // Enter convolution with kernel.
      for (unsigned int x = 0; x < X; ++x) {
        for (unsigned int y = 0; y < Y; ++y) {
          const char m_dx = kernel.at<char>(y, x);
          const char m_dy = kernel.at<char>(x, y);
          __m128i mult_dx = _mm_set_epi16(m_dx, m_dx, m_dx, m_dx, m_dx, m_dx,
                                          m_dx, m_dx);
          __m128i mult_dy = _mm_set_epi16(m_dy, m_dy, m_dy, m_dy, m_dy, m_dy,
                                          m_dy, m_dy);
          unsigned char* p = (src.data + (stride * (i + y)) + x + j);
          __m128i i0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(p));
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
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p_hi_dxdx),
                       _mm_unpackhi_epi16(i_hi_dx_dx, i_lo_dx_dx));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p_lo_dxdx),
                       _mm_unpacklo_epi16(i_hi_dx_dx, i_lo_dx_dx));
      unsigned char* p_lo_dydy = (dydy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j;
      unsigned char* p_hi_dydy = (dydy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j
          + 16;
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p_hi_dydy),
                       _mm_unpackhi_epi16(i_hi_dy_dy, i_lo_dy_dy));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p_lo_dydy),
                       _mm_unpacklo_epi16(i_hi_dy_dy, i_lo_dy_dy));
      unsigned char* p_lo_dxdy = (dxdy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j;
      unsigned char* p_hi_dxdy = (dxdy.data + (2 * stride * (i + cy))) + 2 * cx + 2 * j
          + 16;
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p_hi_dxdy),
                       _mm_unpackhi_epi16(i_hi_dx_dy, i_lo_dx_dy));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p_lo_dxdy),
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

__inline__ void HarrisFeatureDetector::CornerHarris(const agast::Mat& dxdxSmooth,
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
          reinterpret_cast<const __m128i*>(&dxdxSmooth.at<int16_t>(i, j)));
      __m128i dydy = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&dydySmooth.at<int16_t>(i, j)));
      __m128i dxdy = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&dxdySmooth.at<int16_t>(i, j)));

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
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(&dst.at<int>(i, j)), score_1);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(&dst.at<int>(i, j + 4)), score_2);

      // Take care about end.
      j += 8;
      if (j >= maxJ && !end) {
        j = stride - 2 - 8;
        end = true;
      }
    }
  }
}

inline void HarrisFeatureDetector::NonmaxSuppress(
    const agast::Mat& scores, std::vector<agast::KeyPoint>& keypoints) {
  // First do the 8-neighbor nonmax suppression.
  const int stride = scores.cols;
  const int rows_end = scores.rows - 2;
  for (int j = 2; j < rows_end; ++j) {
    const int* p = &scores.at<int>(j, 0);
    const int* const p_begin = p;
    const int* const p_end = &scores.at<int>(j, stride - 2);
    bool last = false;
    while (p < p_end) {
      const int* const center = p;
      ++p;
      if (last) {
        last = false;
        continue;
      }
      if (*center < 64)
        continue;
      if (*(center + 1) > *center)
        continue;
      if (*(center - 1) > *center)
        continue;
      const int* const p1 = (center + stride);
      const int* const p2 = (center - stride);
      if (*p1 > *center)
        continue;
      if (*p2 > *center)
        continue;
      if (*(p1 + 1) > *center)
        continue;
      if (*(p1 - 1) > *center)
        continue;
      if (*(p2 + 1) > *center)
        continue;
      if (*(p2 - 1) > *center)
        continue;
      const int i = p - p_begin;
#if HAVE_OPENCV
      keypoints.push_back(
          agast::KeyPoint(static_cast<float>(i), static_cast<float>(j),
                   10, -1, *center, 0));
#else  // HAVE_OPENCV
      agast::KeyPoint keypoint;
      keypoint.xs = static_cast<float>(i);
      keypoint.ys = static_cast<float>(j);
      keypoint.sigma = 10;
      keypoint.angle = -1;
      keypoint.edge_score = *center;
      keypoint.octave = 0;
      keypoints.push_back(keypoint);
#endif  // HAVE_OPENCV
    }
  }
}

__inline__ void HarrisFeatureDetector::EnforceUniformity(
    const agast::Mat& scores, std::vector<agast::KeyPoint>& keypoints) const {
  // Sort.
  std::sort(keypoints.begin(), keypoints.end(), compareKeypointScore);

  std::vector < agast::KeyPoint > keypoints_new;
  keypoints_new.reserve(keypoints.size());

  // Store occupancy.
  agast::Mat occupancy;
  occupancy = agast::Mat::zeros((scores.rows) / 2 + 32, (scores.cols) / 2 + 32,
                             CV_8U);

  // Go through the sorted keypoints and reject too close ones.
  for (std::vector<agast::KeyPoint>::iterator it = keypoints.begin();
      it != keypoints.end(); ++it) {
    const int cy = (agast::KeyPointX(*it) / 2 + 16);
    const int cx = (agast::KeyPointY(*it) / 2 + 16);

    // Check if this is a high enough score.
    const double s0 = static_cast<double>(occupancy.at<unsigned char>(cy, cx));
    const double s1 = s0 * s0;
    if (static_cast<int16_t>(agast::KeyPointResponse(*it)) < s1 * s1) {
      continue;
    }

    // Masks.
    const float nsc = sqrt(sqrt(agast::KeyPointResponse(*it)));
    for (int y = 0; y < 2 * 16 - 1; ++y) {
      __m128i mem1 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&occupancy.at<unsigned char>(cy + y - 15,
                                                          cx - 15)));
      __m128i mem2 = _mm_loadu_si128(
          reinterpret_cast<__m128i*>(&occupancy.at<unsigned char>(cy + y - 15,
                                                          cx + 1)));
      __m128i mask1 = _mm_set_epi8(_LUT.at<float>(y, 15) * nsc,
                                   _LUT.at<float>(y, 14) * nsc,
                                   _LUT.at<float>(y, 13) * nsc,
                                   _LUT.at<float>(y, 12) * nsc,
                                   _LUT.at<float>(y, 11) * nsc,
                                   _LUT.at<float>(y, 10) * nsc,
                                   _LUT.at<float>(y, 9) * nsc,
                                   _LUT.at<float>(y, 8) * nsc,
                                   _LUT.at<float>(y, 7) * nsc,
                                   _LUT.at<float>(y, 6) * nsc,
                                   _LUT.at<float>(y, 5) * nsc,
                                   _LUT.at<float>(y, 4) * nsc,
                                   _LUT.at<float>(y, 3) * nsc,
                                   _LUT.at<float>(y, 2) * nsc,
                                   _LUT.at<float>(y, 1) * nsc,
                                   _LUT.at<float>(y, 0) * nsc);
      __m128i mask2 = _mm_set_epi8(0, _LUT.at<float>(y, 30) * nsc,
                                   _LUT.at<float>(y, 29) * nsc,
                                   _LUT.at<float>(y, 28) * nsc,
                                   _LUT.at<float>(y, 27) * nsc,
                                   _LUT.at<float>(y, 26) * nsc,
                                   _LUT.at<float>(y, 25) * nsc,
                                   _LUT.at<float>(y, 24) * nsc,
                                   _LUT.at<float>(y, 23) * nsc,
                                   _LUT.at<float>(y, 22) * nsc,
                                   _LUT.at<float>(y, 21) * nsc,
                                   _LUT.at<float>(y, 20) * nsc,
                                   _LUT.at<float>(y, 19) * nsc,
                                   _LUT.at<float>(y, 18) * nsc,
                                   _LUT.at<float>(y, 17) * nsc,
                                   _LUT.at<float>(y, 16) * nsc);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(&occupancy.at<unsigned char>(cy + y - 15,
                                                          cx - 15)),
          _mm_adds_epu8(mem1, mask1));
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(&occupancy.at<unsigned char>(cy + y - 15,
                                                          cx + 1)),
          _mm_adds_epu8(mem2, mask2));
    }

    // Store.
    keypoints_new.push_back(*it);
  }
  keypoints.assign(keypoints_new.begin(), keypoints_new.end());
}
void HarrisFeatureDetector::detectImpl(const agast::Mat& image,
                                       std::vector<agast::KeyPoint>& keypoints,
                                       const agast::Mat& /*mask*/) const {
  keypoints.resize(0);
  agast::Mat scores;
  agast::Mat DxDx1, DyDy1, DxDy1;
  agast::Mat DxDx, DyDy, DxDy;

  // Pipeline.
  GetCovarEntries(image, DxDx1, DyDy1, DxDy1);
  FilterGauss3by316S(DxDx1, DxDx);
  FilterGauss3by316S(DyDy1, DyDy);
  FilterGauss3by316S(DxDy1, DxDy);
  CornerHarris(DxDx, DyDy, DxDy, scores);
  NonmaxSuppress(scores, keypoints);
  EnforceUniformity(scores, keypoints);
}
}  // namespace brisk
#endif  // __ARM_NEON
