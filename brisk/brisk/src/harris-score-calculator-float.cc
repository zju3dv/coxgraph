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

#include <brisk/internal/harris-score-calculator-float.h>
#include <brisk/internal/vectorized-filters.h>

namespace brisk {
void HarrisScoreCalculatorFloat::InitializeScores() {
  agast::Mat DxDx1, DyDy1, DxDy1;
  agast::Mat DxDx, DyDy, DxDy;
  // Pipeline.
  GetCovarEntries(_img, DxDx1, DyDy1, DxDy1);
  FilterGauss3by332F(DxDx1, DxDx);
  FilterGauss3by332F(DyDy1, DyDy);
  FilterGauss3by332F(DxDy1, DxDy);
  CornerHarris(DxDx, DyDy, DxDy, _scores);
}

void HarrisScoreCalculatorFloat::Get2dMaxima(
    float absoluteThreshold, std::vector<PointWithScore>* points) {
  // Do the 8-neighbor nonmax suppression.
  const int stride = _scores.cols;
  const int rows_end = _scores.rows - 2;
  for (int j = 2; j < rows_end; ++j) {
    const float* p = &_scores.at<float>(j, 0);
    const float* const p_begin = p;
    const float* const p_end = &_scores.at<float>(j, stride - 2);
    bool last = false;
    while (p < p_end) {
      const float* const center = p;
      ++p;
      if (last) {
        last = false;
        continue;
      }
      if (*center < absoluteThreshold)
        continue;
      if (*(center + 1) > *center)
        continue;
      if (*(center - 1) > *center)
        continue;
      const float* const p1 = (center + stride);
      const float* const p2 = (center - stride);
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
      const int i = p - p_begin - 1;
#ifdef USE_SIMPLE_POINT_WITH_SCORE
      points->push_back(PointWithScore(*center, i, j));
#else
#error
      points->push_back(PointWithScore(cv::Point2i(i, j), *center));
#endif
    }
  }
}

// X and Y denote the size of the mask.
void HarrisScoreCalculatorFloat::GetCovarEntries(const agast::Mat& src,
                                                 agast::Mat& dxdx, agast::Mat& dydy,
                                                 agast::Mat& dxdy) {
  int jump = 0;  // Number of bytes.
  if (src.type() == CV_8U)
    jump = 1;
  else if (src.type() == CV_16U)
    jump = 2;
  else
    assert(0 && "Unsupported type");

  agast::Mat kernel = agast::Mat::zeros(3, 3, CV_32F);
  kernel.at<float>(0, 0) = 0.09375;
  kernel.at<float>(1, 0) = 0.3125;
  kernel.at<float>(2, 0) = 0.09375;
  kernel.at<float>(0, 2) = -0.09375;
  kernel.at<float>(1, 2) = -0.3125;
  kernel.at<float>(2, 2) = -0.09375;

  const unsigned int X = 3;
  const unsigned int Y = 3;

  // Dest will be floats.
  dxdx = agast::Mat::zeros(src.rows, src.cols, CV_32F);
  dydy = agast::Mat::zeros(src.rows, src.cols, CV_32F);
  dxdy = agast::Mat::zeros(src.rows, src.cols, CV_32F);

  const unsigned int maxJ = ((src.cols - 2) / 4) * 4;
  const unsigned int maxI = src.rows - 2;
  const unsigned int stride = src.cols;

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      __m128 zeros;
      zeros = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);
      __m128 result_dx = zeros;
      __m128 result_dy = zeros;
      // Enter convolution with kernel.
      for (unsigned int x = 0; x < X; ++x) {
        for (unsigned int y = 0; y < Y; ++y) {
          const float m_dx = kernel.at<float>(y, x);
          const float m_dy = kernel.at<float>(x, y);
          __m128 mult_dx = _mm_setr_ps(m_dx, m_dx, m_dx, m_dx);
          __m128 mult_dy = _mm_setr_ps(m_dy, m_dy, m_dy, m_dy);
          __m128 i0;
          if (jump == 1) {
            const unsigned char* p = &src.at < unsigned char > (i + y, x + j);
            i0 = _mm_setr_ps(static_cast<float>(*p),
                             static_cast<float>(*(p + 1)),
                             static_cast<float>(*(p + 2)),
                             static_cast<float>(*(p + 3)));
          } else {
            const uint16_t* p = &src.at < uint16_t > (i + y, x + j);
            i0 = _mm_setr_ps(static_cast<float>(*p),
                             static_cast<float>(*(p + 1)),
                             static_cast<float>(*(p + 2)),
                             static_cast<float>(*(p + 3)));
          }

          if (m_dx != 0) {
            __m128 i_dx = _mm_mul_ps(i0, mult_dx);
            result_dx = _mm_add_ps(result_dx, i_dx);
          }

          if (m_dy != 0) {
            __m128 i_dy = _mm_mul_ps(i0, mult_dy);
            result_dy = _mm_add_ps(result_dy, i_dy);
          }
        }
      }

      // Calculate covariance entries - remove precision (ends up being 4 bit),
      // then remove 4 more bits.
      __m128 i_dx_dx = _mm_mul_ps(result_dx, result_dx);
      __m128 i_dx_dy = _mm_mul_ps(result_dy, result_dx);
      __m128 i_dy_dy = _mm_mul_ps(result_dy, result_dy);

      // Store.
      _mm_storeu_ps(&dxdx.at<float>(i, j + 1), i_dx_dx);
      _mm_storeu_ps(&dxdy.at<float>(i, j + 1), i_dx_dy);
      _mm_storeu_ps(&dydy.at<float>(i, j + 1), i_dy_dy);

      // Take care about end.
      j += 4;
      if (j >= maxJ && !end) {
        j = stride - 2 - 4;
        end = true;
      }
    }
  }
}

void HarrisScoreCalculatorFloat::CornerHarris(const agast::Mat& dxdxSmooth,
                                              const agast::Mat& dydySmooth,
                                              const agast::Mat& dxdySmooth,
                                              agast::Mat& dst) {
  // Dest will be float.
  dst = agast::Mat::zeros(dxdxSmooth.rows, dxdxSmooth.cols, CV_32F);
  const unsigned int maxJ = ((dxdxSmooth.cols - 2) / 8) * 8;
  const unsigned int maxI = dxdxSmooth.rows - 2;
  const unsigned int stride = dxdxSmooth.cols;

  for (unsigned int i = 0; i < maxI; ++i) {
    bool end = false;
    for (unsigned int j = 0; j < maxJ;) {
      __m128 dxdx = _mm_loadu_ps(&dxdxSmooth.at<float>(i, j));
      __m128 dydy = _mm_loadu_ps(&dydySmooth.at<float>(i, j));
      __m128 dxdy = _mm_loadu_ps(&dxdySmooth.at<float>(i, j));

      // Determinant terms.
      __m128 prod1 = _mm_mul_ps(dxdx, dydy);
      __m128 prod2 = _mm_mul_ps(dxdy, dxdy);

      // Calculate the determinant.
      __m128 det = _mm_sub_ps(prod1, prod2);

      // Trace - uses kappa = 1 / 16.
      __m128 trace = _mm_add_ps(dxdx, dydy);
      __m128 trace_sq = _mm_mul_ps(trace, trace);
      __m128 trace_sq_00625 = _mm_mul_ps(
          trace_sq, _mm_setr_ps(0.0625, 0.0625, 0.0625, 0.0625));

      // Form score.
      __m128 score = _mm_sub_ps(det, trace_sq_00625);

      // Store.
      _mm_storeu_ps(&dst.at<float>(i, j), score);

      // Take care about end.
      j += 4;
      if (j >= maxJ && !end) {
        j = stride - 2 - 4;
        end = true;
      }
    }
  }
}
}  // namespace brisk
#endif  // __ARM_NEON
