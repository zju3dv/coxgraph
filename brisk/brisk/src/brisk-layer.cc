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

#include <brisk/internal/brisk-layer.h>
#include <brisk/internal/image-down-sampling.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#include <emmintrin.h>
#include <tmmintrin.h>
#endif  // __ARM_NEON

namespace brisk {
// Construct a layer.
BriskLayer::BriskLayer(const agast::Mat& img, unsigned char upperThreshold,
                       unsigned char lowerThreshold, float scale, float offset) {
  upperThreshold_ = upperThreshold;
  lowerThreshold_ = lowerThreshold;

  img_ = img;
  scores_ = agast::Mat::zeros(img.rows, img.cols, CV_8U);
  // Attention: this means that the passed image reference must point to
  // persistent memory.
  scale_ = scale;
  offset_ = offset;
  // Create an agast detector.
  oastDetector_.reset(new agast::OastDetector9_16(img.cols, img.rows, 0));
  agastDetector_5_8_.reset(new agast::AgastDetector5_8(img.cols, img.rows, 0));

  // Calculate threshold map.
  CalculateThresholdMap();
}
// Derive a layer.
BriskLayer::BriskLayer(const BriskLayer& layer, int mode, unsigned char upperThreshold,
                       unsigned char lowerThreshold) {
  upperThreshold_ = upperThreshold;
  lowerThreshold_ = lowerThreshold;

  if (mode == CommonParams::HALFSAMPLE) {
    img_.create(layer.img().rows / 2, layer.img().cols / 2, CV_8U);
    Halfsample8(layer.img(), img_);
    scale_ = layer.scale() * 2;
    offset_ = 0.5 * scale_ - 0.5;
  } else {
    img_.create(2 * (layer.img().rows / 3), 2 * (layer.img().cols / 3), CV_8U);
    Twothirdsample8(layer.img(), img_);
    scale_ = layer.scale() * 1.5;
    offset_ = 0.5 * scale_ - 0.5;
  }
  scores_ = agast::Mat::zeros(img_.rows, img_.cols, CV_8U);
  oastDetector_.reset(new agast::OastDetector9_16(img_.cols, img_.rows, 0));
  agastDetector_5_8_.reset(
      new agast::AgastDetector5_8(img_.cols, img_.rows, 0));

  // Calculate threshold map.
  CalculateThresholdMap();
}

// Fast/Agast.
// Wraps the agast class.
void BriskLayer::GetAgastPoints(uint8_t threshold,
                                std::vector<agast::KeyPoint>* keypoints) {
  CHECK_NOTNULL(keypoints);
  oastDetector_->set_threshold(threshold, upperThreshold_, lowerThreshold_);
  if (keypoints->empty()) {
    oastDetector_->detect(img_.data, *keypoints, &thrmap_);
  }
  // Also write scores.
  const int num = keypoints->size();
  const int imcols = img_.cols;

  for (int i = 0; i < num; i++) {
    const int offs = agast::KeyPointX((*keypoints)[i]) +
        agast::KeyPointY((*keypoints)[i]) * imcols;
    int thr = *(thrmap_.data + offs);
    oastDetector_->set_threshold(thr);
    *(scores_.data + offs) = oastDetector_->cornerScore(img_.data + offs);
  }
}
uint8_t BriskLayer::GetAgastScore(int x, int y, uint8_t threshold) {
  if (x < 3 || y < 3)
    return 0;
  if (x >= img_.cols - 3 || y >= img_.rows - 3)
    return 0;
  uint8_t& score = *(scores_.data + x + y * scores_.cols);
  if (score > 2) {
    return score;
  }
  oastDetector_->set_threshold(threshold - 1);
  score = oastDetector_->cornerScore(img_.data + x + y * img_.cols);
  if (score < threshold)
    score = 0;
  return score;
}

uint8_t BriskLayer::GetAgastScore_5_8(int x, int y, uint8_t threshold) {
  if (x < 2 || y < 2)
    return 0;
  if (x >= img_.cols - 2 || y >= img_.rows - 2)
    return 0;
  agastDetector_5_8_->set_threshold(threshold - 1);
  uint8_t score = agastDetector_5_8_->cornerScore(
      img_.data + x + y * img_.cols);
  if (score < threshold)
    score = 0;
  return score;
}

uint8_t BriskLayer::GetAgastScore(float xf, float yf, uint8_t threshold,
                                  float scale) {
  if (scale <= 1.0f) {
    // Just do an interpolation inside the layer.
    const int x = static_cast<int>(xf);
    const float rx1 = xf - static_cast<float>(x);
    const float rx = 1.0f - rx1;
    const int y = static_cast<int>(yf);
    const float ry1 = yf - static_cast<float>(y);
    const float ry = 1.0f - ry1;

    return rx * ry * GetAgastScore(x, y, threshold)
        + rx1 * ry * GetAgastScore(x + 1, y, threshold)
        + rx * ry1 * GetAgastScore(x, y + 1, threshold)
        + rx1 * ry1 * GetAgastScore(x + 1, y + 1, threshold);
  } else {
    // This means we overlap area smoothing.
    const float halfscale = scale / 2.0f;
    // Get the scores first:
    for (int x = static_cast<int>(xf - halfscale);
        x <= static_cast<int>(xf + halfscale + 1.0f); x++) {
      for (int y = static_cast<int>(yf - halfscale);
          y <= static_cast<int>(yf + halfscale + 1.0f); y++) {
        GetAgastScore(x, y, threshold);
      }
    }
    // Get the smoothed value.
    return Value(scores_, xf, yf, scale);
  }
}

// Access gray values (smoothed/interpolated).
uint8_t BriskLayer::Value(const agast::Mat& mat, float xf, float yf, float scale) {
  assert(!mat.empty());
  // Get the position.
  const int x = floor(xf);
  const int y = floor(yf);
  const agast::Mat& image = mat;
  const int& imagecols = image.cols;

  // Get the sigma_half:
  const float sigma_half = scale / 2;
  const float area = 4.0 * sigma_half * sigma_half;
  // Calculate output:
  int ret_val;
  if (sigma_half < 0.5) {
    // Interpolation multipliers:
    const int r_x = (xf - x) * 1024;
    const int r_y = (yf - y) * 1024;
    const int r_x_1 = (1024 - r_x);
    const int r_y_1 = (1024 - r_y);
    unsigned char* ptr = image.data + x + y * imagecols;
    // Just interpolate:
    ret_val = (r_x_1 * r_y_1 * static_cast<int>(*ptr));
    ptr++;
    ret_val += (r_x * r_y_1 * static_cast<int>(*ptr));
    ptr += imagecols;
    ret_val += (r_x * r_y * static_cast<int>(*ptr));
    ptr--;
    ret_val += (r_x_1 * r_y * static_cast<int>(*ptr));
    return 0xFF & ((ret_val + 512) / 1024 / 1024);
  }

  // This is the standard case (simple, not speed optimized yet):

  // Scaling:
  const int scaling = 4194304.0 / area;
  const int scaling2 = static_cast<float>(scaling) * area / 1024.0;

  // Calculate borders.
  const float x_1 = xf - sigma_half;
  const float x1 = xf + sigma_half;
  const float y_1 = yf - sigma_half;
  const float y1 = yf + sigma_half;

  const int x_left = static_cast<int>(x_1 + 0.5);
  const int y_top = static_cast<int>(y_1 + 0.5);
  const int x_right = static_cast<int>(x1 + 0.5);
  const int y_bottom = static_cast<int>(y1 + 0.5);

  // Overlap area - multiplication factors:
  const float r_x_1 = static_cast<float>(x_left) - x_1 + 0.5;
  const float r_y_1 = static_cast<float>(y_top) - y_1 + 0.5;
  const float r_x1 = x1 - static_cast<float>(x_right) + 0.5;
  const float r_y1 = y1 - static_cast<float>(y_bottom) + 0.5;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (r_x_1 * r_y_1) * scaling;
  const int B = (r_x1 * r_y_1) * scaling;
  const int C = (r_x1 * r_y1) * scaling;
  const int D = (r_x_1 * r_y1) * scaling;
  const int r_x_1_i = r_x_1 * scaling;
  const int r_y_1_i = r_y_1 * scaling;
  const int r_x1_i = r_x1 * scaling;
  const int r_y1_i = r_y1 * scaling;

  // Now the calculation:
  unsigned char* ptr = image.data + x_left + imagecols * y_top;
  // First row:
  ret_val = A * static_cast<int>(*ptr);
  ptr++;
  const unsigned char* end1 = ptr + dx;
  for (; ptr < end1; ptr++) {
    ret_val += r_y_1_i * static_cast<int>(*ptr);
  }
  ret_val += B * static_cast<int>(*ptr);
  // Middle ones:
  ptr += imagecols - dx - 1;
  unsigned char* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1) {
    ret_val += r_x_1_i * static_cast<int>(*ptr);
    ptr++;
    const unsigned char* end2 = ptr + dx;
    for (; ptr < end2; ptr++) {
      ret_val += static_cast<int>(*ptr) * scaling;
    }
    ret_val += r_x1_i * static_cast<int>(*ptr);
  }
  // Last row:
  ret_val += D * static_cast<int>(*ptr);
  ptr++;
  const unsigned char* end3 = ptr + dx;
  for (; ptr < end3; ptr++) {
    ret_val += r_y1_i * static_cast<int>(*ptr);
  }
  ret_val += C * static_cast<int>(*ptr);

  return 0xFF & ((ret_val + scaling2 / 2) / scaling2 / 1024);
}

// Threshold map.
void BriskLayer::CalculateThresholdMap() {
  // Allocate threshold map.
  agast::Mat tmpmax = agast::Mat::zeros(img_.rows, img_.cols, CV_8U);
  agast::Mat tmpmin = agast::Mat::zeros(img_.rows, img_.cols, CV_8U);
  thrmap_ = agast::Mat::zeros(img_.rows, img_.cols, CV_8U);

  const int rowstride = img_.cols;

  for (int y = 1; y < img_.rows - 1; y++) {
    int x = 1;
    while (x + 16 < img_.cols - 1) {
      // Access.
      unsigned char* p = img_.data + x - 1 + (y - 1) * rowstride;
#ifdef __ARM_NEON
      // NEON version.
      uint8x16_t v_1_1 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p++;
      uint8x16_t v0_1 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p++;
      uint8x16_t v1_1 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += rowstride;
      uint8x16_t v10 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p--;
      uint8x16_t v00 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p--;
      uint8x16_t v_10 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += rowstride;
      uint8x16_t v_11 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p++;
      uint8x16_t v01 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p++;
      uint8x16_t v11 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));

      // Min/max calculation.
      uint8x16_t max = vmaxq_u8(v_1_1, v0_1);
      uint8x16_t min = vminq_u8(v_1_1, v0_1);
      max = vmaxq_u8(max, v1_1);
      min = vminq_u8(min, v1_1);
      max = vmaxq_u8(max, v10);
      min = vminq_u8(min, v10);
      max = vmaxq_u8(max, v00);
      min = vminq_u8(min, v00);
      max = vmaxq_u8(max, v_10);
      min = vminq_u8(min, v_10);
      max = vmaxq_u8(max, v_11);
      min = vminq_u8(min, v_11);
      max = vmaxq_u8(max, v01);
      min = vminq_u8(min, v01);
      max = vmaxq_u8(max, v11);
      min = vminq_u8(min, v11);

      // Store data back:
      vst1q_u8(reinterpret_cast<uint8_t*>(tmpmax.data + x + y * rowstride), max);
      vst1q_u8(reinterpret_cast<uint8_t*>(tmpmin.data + x + y * rowstride), min);
#else
      // SSE version.
      __m128i v_1_1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p++;
      __m128i v0_1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p++;
      __m128i v1_1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += rowstride;
      __m128i v10 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p--;
      __m128i v00 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p--;
      __m128i v_10 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += rowstride;
      __m128i v_11 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p++;
      __m128i v01 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p++;
      __m128i v11 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));

      // Min/max calc.
      __m128i max = _mm_max_epu8(v_1_1, v0_1);
      __m128i min = _mm_min_epu8(v_1_1, v0_1);
      max = _mm_max_epu8(max, v1_1);
      min = _mm_min_epu8(min, v1_1);
      max = _mm_max_epu8(max, v10);
      min = _mm_min_epu8(min, v10);
      max = _mm_max_epu8(max, v00);
      min = _mm_min_epu8(min, v00);
      max = _mm_max_epu8(max, v_10);
      min = _mm_min_epu8(min, v_10);
      max = _mm_max_epu8(max, v_11);
      min = _mm_min_epu8(min, v_11);
      max = _mm_max_epu8(max, v01);
      min = _mm_min_epu8(min, v01);
      max = _mm_max_epu8(max, v11);
      min = _mm_min_epu8(min, v11);

      // Store.
      _mm_storeu_si128(
          reinterpret_cast<__m128i *>(tmpmax.data + x + y * rowstride), max);
      _mm_storeu_si128(
          reinterpret_cast<__m128i *>(tmpmin.data + x + y * rowstride), min);
#endif  // __ARM_NEON
      // Next block.
      x += 16;
    }
  }

  for (int y = 3; y < img_.rows - 3; y++) {
    int x = 3;
    while (x + 16 < img_.cols - 3) {
      // Access.
      unsigned char* p = img_.data + x + y * rowstride;
#ifdef __ARM_NEON
      // NEON version //
      uint8x16_t v00 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p -= 2 + 2 * rowstride;
      uint8x16_t v_2_2 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += 4;
      uint8x16_t v2_2 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += 4 * rowstride;
      uint8x16_t v22 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p -= 4;
      uint8x16_t v_22 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));

      p = tmpmax.data + x + (y - 2) * rowstride;
      uint8x16_t max0_2 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += 4 * rowstride;
      uint8x16_t max02 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p -= 2 * rowstride + 2;
      uint8x16_t max_20 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += 4;
      uint8x16_t max20 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));

      p = tmpmin.data + x + (y - 2) * rowstride;
      uint8x16_t min0_2 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += 4 * rowstride;
      uint8x16_t min02 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p -= 2 * rowstride + 2;
      uint8x16_t min_20 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
      p += 4;
      uint8x16_t min20 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));

      // Min/max.
      uint8x16_t max = vmaxq_u8(v00, v_2_2);
      uint8x16_t min = vminq_u8(v00, v_2_2);
      max = vmaxq_u8(max, v2_2);
      min = vminq_u8(min, v2_2);
      max = vmaxq_u8(max, v22);
      min = vminq_u8(min, v22);
      max = vmaxq_u8(max, v_22);
      min = vminq_u8(min, v_22);
      max = vmaxq_u8(max, max0_2);
      min = vminq_u8(min, min0_2);
      max = vmaxq_u8(max, max02);
      min = vminq_u8(min, min02);
      max = vmaxq_u8(max, max_20);
      min = vminq_u8(min, min_20);
      max = vmaxq_u8(max, max20);
      min = vminq_u8(min, min20);

      // Store the data back:
      uint8x16_t diff = vsubq_u8(max, min);
      vst1q_u8(reinterpret_cast<uint8_t*> (thrmap_.data + x + y * rowstride),
               diff);
#else
      // SSE version.
      __m128i v00 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p -= 2 + 2 * rowstride;
      __m128i v_2_2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += 4;
      __m128i v2_2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += 4 * rowstride;
      __m128i v22 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p -= 4;
      __m128i v_22 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));

      p = tmpmax.data + x + (y - 2) * rowstride;
      __m128i max0_2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += 4 * rowstride;
      __m128i max02 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p -= 2 * rowstride + 2;
      __m128i max_20 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += 4;
      __m128i max20 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));

      p = tmpmin.data + x + (y - 2) * rowstride;
      __m128i min0_2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += 4 * rowstride;
      __m128i min02 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p -= 2 * rowstride + 2;
      __m128i min_20 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));
      p += 4;
      __m128i min20 = _mm_loadu_si128(reinterpret_cast<__m128i *>(p));

      // Min / max.
      __m128i max = _mm_max_epu8(v00, v_2_2);
      __m128i min = _mm_min_epu8(v00, v_2_2);
      max = _mm_max_epu8(max, v2_2);
      min = _mm_min_epu8(min, v2_2);
      max = _mm_max_epu8(max, v22);
      min = _mm_min_epu8(min, v22);
      max = _mm_max_epu8(max, v_22);
      min = _mm_min_epu8(min, v_22);
      max = _mm_max_epu8(max, max0_2);
      min = _mm_min_epu8(min, min0_2);
      max = _mm_max_epu8(max, max02);
      min = _mm_min_epu8(min, min02);
      max = _mm_max_epu8(max, max_20);
      min = _mm_min_epu8(min, min_20);
      max = _mm_max_epu8(max, max20);
      min = _mm_min_epu8(min, min20);

      // Store.
      __m128i diff = _mm_sub_epi8(max, min);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(thrmap_.data + x +
          y * rowstride), diff);
#endif  // __ARM_NEON
      // Next block.
      x += 16;
    }
  }

  for (int x = std::max(1, 16 * ((img_.cols - 2) / 16) - 16); x < img_.cols - 1;
      x++) {
    for (int y = 1; y < img_.rows - 1; y++) {
      // Access.
      unsigned char* p = img_.data + x - 1 + (y - 1) * rowstride;
      int v_1_1 = *p;
      p++;
      int v0_1 = *p;
      p++;
      int v1_1 = *p;
      p += rowstride;
      int v10 = *p;
      p--;
      int v00 = *p;
      p--;
      int v_10 = *p;
      p += rowstride;
      int v_11 = *p;
      p++;
      int v01 = *p;
      p++;
      int v11 = *p;

      // Min/max calc.
      int max = std::max(v_1_1, v0_1);
      int min = std::min(v_1_1, v0_1);
      max = std::max(max, v1_1);
      min = std::min(min, v1_1);
      max = std::max(max, v10);
      min = std::min(min, v10);
      max = std::max(max, v00);
      min = std::min(min, v00);
      max = std::max(max, v_10);
      min = std::min(min, v_10);
      max = std::max(max, v_11);
      min = std::min(min, v_11);
      max = std::max(max, v01);
      min = std::min(min, v01);
      max = std::max(max, v11);
      min = std::min(min, v11);

      // Store.
      *(tmpmax.data + x + y * rowstride) = max;
      *(tmpmin.data + x + y * rowstride) = min;
    }
  }

  for (int x = std::max(3, 16 * ((img_.cols - 6) / 16) - 16); x < img_.cols - 3;
      x++) {
    for (int y = 3; y < img_.rows - 3; y++) {
      // Access.
      unsigned char* p = img_.data + x + y * rowstride;
      int v00 = *p;
      p -= 2 + 2 * rowstride;
      int v_2_2 = *p;
      p += 4;
      int v2_2 = *p;
      p += 4 * rowstride;
      int v22 = *p;
      p -= 4;
      int v_22 = *p;

      p = tmpmax.data + x + (y - 2) * rowstride;
      int max0_2 = *p;
      p += 4 * rowstride;
      int max02 = *p;
      p -= 2 * rowstride + 2;
      int max_20 = *p;
      p += 4;
      int max20 = *p;

      p = tmpmin.data + x + (y - 2) * rowstride;
      int min0_2 = *p;
      p += 4 * rowstride;
      int min02 = *p;
      p -= 2 * rowstride + 2;
      int min_20 = *p;
      p += 4;
      int min20 = *p;

      // Min / max.
      int max = std::max(v00, v_2_2);
      int min = std::min(v00, v_2_2);
      max = std::max(max, v2_2);
      min = std::min(min, v2_2);
      max = std::max(max, v22);
      min = std::min(min, v22);
      max = std::max(max, v_22);
      min = std::min(min, v_22);
      max = std::max(max, max0_2);
      min = std::min(min, min0_2);
      max = std::max(max, max02);
      min = std::min(min, min02);
      max = std::max(max, max_20);
      min = std::min(min, min_20);
      max = std::max(max, max20);
      min = std::min(min, min20);

      // Store.
      *(thrmap_.data + x + y * rowstride) = max - min;
    }
  }
}
}  // namespace brisk
