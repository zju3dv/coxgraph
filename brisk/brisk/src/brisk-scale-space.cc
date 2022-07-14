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
#include <brisk/internal/brisk-scale-space.h>

namespace brisk {
const float BriskScaleSpace::kBasicSize_ = 12.0;

const int BriskScaleSpace::kMaxThreshold_ = 1;
const int BriskScaleSpace::kDropThreshold_ = 5;
const int BriskScaleSpace::kMinDrop_ = 15;
const unsigned char BriskScaleSpace::kDefaultLowerThreshold = 10;  // Originally 28.
const unsigned char BriskScaleSpace::kDefaultUpperThreshold = 230;

// Construct telling the octaves number:
BriskScaleSpace::BriskScaleSpace(uint8_t _octaves,
                                 bool suppressScaleNonmaxima) {
  suppressScaleNonmaxima_ = suppressScaleNonmaxima;
  if (_octaves == 0)
    layers_ = 1;
  else
    layers_ = 2 * _octaves;
}
BriskScaleSpace::~BriskScaleSpace() { }
// Construct the image pyramids.
void BriskScaleSpace::ConstructPyramid(const agast::Mat& image, unsigned char threshold,
                                       unsigned char overwrite_lower_thres) {
  // Set correct size:
  pyramid_.clear();

  // Assign threshold.
  threshold_ = threshold;

  // Fill the pyramid:
  pyramid_.push_back(
      BriskLayer(image.clone(), kDefaultUpperThreshold, overwrite_lower_thres));
  if (layers_ > 1) {
    pyramid_.push_back(
        BriskLayer(pyramid_.back(), BriskLayer::CommonParams::TWOTHIRDSAMPLE,
                   (kDefaultUpperThreshold), (overwrite_lower_thres)));
  }
  const int octaves2 = layers_;

  for (uint8_t i = 2; i < octaves2; i += 2) {
    pyramid_.push_back(
        BriskLayer(pyramid_[i - 2], BriskLayer::CommonParams::HALFSAMPLE,
                   (kDefaultUpperThreshold), (overwrite_lower_thres)));
    pyramid_.push_back(
        BriskLayer(pyramid_[i - 1], BriskLayer::CommonParams::HALFSAMPLE,
                   (kDefaultUpperThreshold), (overwrite_lower_thres)));
  }
}

void BriskScaleSpace::GetKeypoints(std::vector<agast::KeyPoint>* keypoints) {
  CHECK_NOTNULL(keypoints);
  std::vector<std::vector<agast::KeyPoint> > agastPoints;
  agastPoints.resize(layers_);

  bool perform_2d_nonMax = true;

  // Go through the octaves and intra layers and calculate fast corner scores:
  for (uint8_t i = 0; i < layers_; ++i) {
    BriskLayer& l = pyramid_[i];

    // Compute scores for given keypoints or extract new kepoints.
    if (!keypoints->empty()) {
      perform_2d_nonMax = false;
      // Compute the location for this layer:
      for (const agast::KeyPoint& keypoint : *keypoints) {
        agast::KeyPoint kp = keypoint;
        agast::KeyPointX(kp) =
            (static_cast<float>(agast::KeyPointX(keypoint))) /
            l.scale() - l.offset();
        agast::KeyPointY(kp) =
            (static_cast<float>(agast::KeyPointY(keypoint))) /
            l.scale() - l.offset();
        if (agast::KeyPointX(kp) < 3 || agast::KeyPointY(kp) < 3 ||
            agast::KeyPointX(kp) > l.cols() - 3 ||
            agast::KeyPointY(kp) > l.rows() - 3) {
          continue;
        }
        // This calculates and stores the score of this keypoint in the score map.
        l.GetAgastScore(agast::KeyPointX(kp), agast::KeyPointY(kp), 0);
        agastPoints.at(i).push_back(kp);
      }
    }

    l.GetAgastPoints(threshold_, &agastPoints[i]);
  }

  keypoints->clear();

  if (!suppressScaleNonmaxima_) {
    for (uint8_t i = 0; i < layers_; i++) {
      // Just do a simple 2d subpixel refinement...
      const int num = agastPoints[i].size();
      for (int n = 0; n < num; n++) {
        const agast::KeyPoint& keypoint = agastPoints.at(0)[n];
        const float& point_x = agast::KeyPointX(keypoint);
        const float& point_y = agast::KeyPointY(keypoint);
        // First check if it is a maximum:
        if (perform_2d_nonMax && !IsMax2D(i, point_x, point_y))
          continue;

        // Let's do the subpixel and float scale refinement:
        brisk::BriskLayer& l = pyramid_[i];
        int s_0_0 = l.GetAgastScore(point_x - 1, point_y - 1, 1);
        int s_1_0 = l.GetAgastScore(point_x, point_y - 1, 1);
        int s_2_0 = l.GetAgastScore(point_x + 1, point_y - 1, 1);
        int s_2_1 = l.GetAgastScore(point_x + 1, point_y, 1);
        int s_1_1 = l.GetAgastScore(point_x, point_y, 1);
        int s_0_1 = l.GetAgastScore(point_x - 1, point_y, 1);
        int s_0_2 = l.GetAgastScore(point_x - 1, point_y + 1, 1);
        int s_1_2 = l.GetAgastScore(point_x, point_y + 1, 1);
        int s_2_2 = l.GetAgastScore(point_x + 1, point_y + 1, 1);
        float delta_x, delta_y;
        float max = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0,
                               s_2_1, s_2_2, delta_x, delta_y);

        // Store:
        agast::KeyPoint kp = keypoint;
        agast::KeyPointX(kp) = static_cast<float>(point_x) + delta_x;
        agast::KeyPointY(kp) = static_cast<float>(point_y) + delta_y;
        agast::KeyPointSize(kp) = kBasicSize_ * l.scale();
        agast::KeyPointAngle(kp) = -1;
        agast::KeyPointResponse(kp) = max;
        agast::KeyPointOctave(kp) = 0;
        keypoints->push_back(kp);
      }
    }
    return;
  }

  if (layers_ == 1) {
    // Just do a simple 2d subpixel refinement...
    const int num = agastPoints[0].size();
    for (int n = 0; n < num; n++) {
      const agast::KeyPoint& keypoint = agastPoints.at(0)[n];
      const float& point_x = agast::KeyPointX(keypoint);
      const float& point_y = agast::KeyPointY(keypoint);

      // First check if it is a maximum:
      if (perform_2d_nonMax && !IsMax2D(0, point_x, point_y))
        continue;

      // Let's do the subpixel and float scale refinement:
      brisk::BriskLayer& l = pyramid_[0];
      int s_0_0 = l.GetAgastScore(point_x - 1, point_y - 1, 1);
      int s_1_0 = l.GetAgastScore(point_x, point_y - 1, 1);
      int s_2_0 = l.GetAgastScore(point_x + 1, point_y - 1, 1);
      int s_2_1 = l.GetAgastScore(point_x + 1, point_y, 1);
      int s_1_1 = l.GetAgastScore(point_x, point_y, 1);
      int s_0_1 = l.GetAgastScore(point_x - 1, point_y, 1);
      int s_0_2 = l.GetAgastScore(point_x - 1, point_y + 1, 1);
      int s_1_2 = l.GetAgastScore(point_x, point_y + 1, 1);
      int s_2_2 = l.GetAgastScore(point_x + 1, point_y + 1, 1);
      float delta_x, delta_y;
      float max = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0,
                             s_2_1, s_2_2, delta_x, delta_y);
      // Store:
      agast::KeyPoint kp = keypoint;
      agast::KeyPointX(kp) = static_cast<float>(point_x) + delta_x;
      agast::KeyPointY(kp) = static_cast<float>(point_y) + delta_y;
      agast::KeyPointSize(kp) = kBasicSize_;
      agast::KeyPointAngle(kp) = -1;
      agast::KeyPointResponse(kp) = max;
      agast::KeyPointOctave(kp) = 0;
      keypoints->push_back(kp);
    }
    return;
  }

  float x, y, scale, score;
  for (uint8_t i = 0; i < layers_; i++) {
    brisk::BriskLayer& l = pyramid_[i];
    const int num = agastPoints[i].size();
    if (i == layers_ - 1) {
      for (int n = 0; n < num; n++) {
        const agast::KeyPoint& keypoint = agastPoints.at(i)[n];
        const float& point_x = agast::KeyPointX(keypoint);
        const float& point_y = agast::KeyPointY(keypoint);
        // Consider only 2D maxima...
        if (perform_2d_nonMax && !IsMax2D(i, point_x, point_y))
          continue;

        bool ismax;
        float dx, dy;
        GetScoreMaxBelow(i, point_x, point_y,
                         l.GetAgastScore(point_x, point_y, 1), ismax, dx, dy);
        if (!ismax)
          continue;

        // Get the patch on this layer:
        int s_0_0 = l.GetAgastScore(point_x - 1, point_y - 1, 1);
        int s_1_0 = l.GetAgastScore(point_x, point_y - 1, 1);
        int s_2_0 = l.GetAgastScore(point_x + 1, point_y - 1, 1);
        int s_2_1 = l.GetAgastScore(point_x + 1, point_y, 1);
        int s_1_1 = l.GetAgastScore(point_x, point_y, 1);
        int s_0_1 = l.GetAgastScore(point_x - 1, point_y, 1);
        int s_0_2 = l.GetAgastScore(point_x - 1, point_y + 1, 1);
        int s_1_2 = l.GetAgastScore(point_x, point_y + 1, 1);
        int s_2_2 = l.GetAgastScore(point_x + 1, point_y + 1, 1);
        float delta_x, delta_y;
        float max = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0,
                               s_2_1, s_2_2, delta_x, delta_y);

        // Store:
        agast::KeyPoint kp = keypoint;
        agast::KeyPointX(kp) = (static_cast<float>(point_x) + delta_x) *
            l.scale() + l.offset();
        agast::KeyPointY(kp) = (static_cast<float>(point_y) + delta_y) *
            l.scale() + l.offset();
        agast::KeyPointSize(kp) = kBasicSize_ * l.scale();
        agast::KeyPointAngle(kp) = -1;
        agast::KeyPointResponse(kp) = max;
        agast::KeyPointOctave(kp) = i;
        keypoints->push_back(kp);
      }
    } else {
      // Not the last layer:
      for (int n = 0; n < num; n++) {
        const agast::KeyPoint& keypoint = agastPoints.at(i)[n];
        const float& point_x = agast::KeyPointX(keypoint);
        const float& point_y = agast::KeyPointY(keypoint);

        // First check if it is a maximum:
        if (perform_2d_nonMax && !IsMax2D(i, point_x, point_y))
          continue;

        // Let's do the subpixel and float scale refinement:
        bool ismax;
        score = Refine3D(i, point_x, point_y, x, y, scale, ismax);
        if (!ismax) {
          continue;
        }

        // Finally store the detected keypoint:
        agast::KeyPoint kp = keypoint;
        agast::KeyPointX(kp) = x;
        agast::KeyPointY(kp) = y;
        agast::KeyPointSize(kp) = kBasicSize_ * scale;
        agast::KeyPointAngle(kp) = -1;
        agast::KeyPointResponse(kp) = score;
        agast::KeyPointOctave(kp) = i;
        keypoints->push_back(kp);
      }
    }
  }
}

// Interpolated score access with recalculation when needed:
__inline__ int BriskScaleSpace::GetScoreAbove(const uint8_t layer,
                                              const int x_layer,
                                              const int y_layer) {
  assert(layer < layers_ - 1);
  brisk::BriskLayer& l = pyramid_[layer + 1];
  if (layer % 2 == 0) {  // Octave.
    const int sixths_x = 4 * x_layer - 1;
    const int x_above = sixths_x / 6;
    const int sixths_y = 4 * y_layer - 1;
    const int y_above = sixths_y / 6;
    const int r_x = (sixths_x % 6);
    const int r_x_1 = 6 - r_x;
    const int r_y = (sixths_y % 6);
    const int r_y_1 = 6 - r_y;
    uint8_t score = 0xFF
        & ((r_x_1 * r_y_1 * l.GetAgastScore(x_above, y_above, 1)
            + r_x * r_y_1 * l.GetAgastScore(x_above + 1, y_above, 1)
            + r_x_1 * r_y * l.GetAgastScore(x_above, y_above + 1, 1)
            + r_x * r_y * l.GetAgastScore(x_above + 1, y_above + 1, 1) + 18)
            / 36);

    return score;
  } else {  // Intra layers.
    const int eighths_x = 6 * x_layer - 1;
    const int x_above = eighths_x / 8;
    const int eighths_y = 6 * y_layer - 1;
    const int y_above = eighths_y / 8;
    const int r_x = (eighths_x % 8);
    const int r_x_1 = 8 - r_x;
    const int r_y = (eighths_y % 8);
    const int r_y_1 = 8 - r_y;
    uint8_t score = 0xFF
        & ((r_x_1 * r_y_1 * l.GetAgastScore(x_above, y_above, 1)
            + r_x * r_y_1 * l.GetAgastScore(x_above + 1, y_above, 1)
            + r_x_1 * r_y * l.GetAgastScore(x_above, y_above + 1, 1)
            + r_x * r_y * l.GetAgastScore(x_above + 1, y_above + 1, 1) + 32)
            / 64);
    return score;
  }
}
__inline__ int BriskScaleSpace::GetScoreBelow(const uint8_t layer,
                                              const int x_layer,
                                              const int y_layer) {
  assert(layer);
  brisk::BriskLayer& l = pyramid_[layer - 1];
  int sixth_x;
  int quarter_x;
  float xf;
  int sixth_y;
  int quarter_y;
  float yf;

  // Scaling:
  float offs;
  float area;
  int scaling;
  int scaling2;

  if (layer % 2 == 0) {  // Octave.
    sixth_x = 8 * x_layer + 1;
    xf = static_cast<float>(sixth_x) / 6.0;
    sixth_y = 8 * y_layer + 1;
    yf = static_cast<float>(sixth_y) / 6.0;

    // Scaling:
    offs = 3.0 / 4.0;
    area = 4.0 * offs * offs;
    scaling = 4194304.0 / area;
    scaling2 = static_cast<float>(scaling) * area;
  } else {
    quarter_x = 6 * x_layer + 1;
    xf = static_cast<float>(quarter_x) / 4.0;
    quarter_y = 6 * y_layer + 1;
    yf = static_cast<float>(quarter_y) / 4.0;

    // Scaling:
    offs = 2.0 / 3.0;
    area = 4.0 * offs * offs;
    scaling = 4194304.0 / area;
    scaling2 = static_cast<float>(scaling) * area;
  }

  // Calculate borders.
  const float x_1 = xf - offs;
  const float x1 = xf + offs;
  const float y_1 = yf - offs;
  const float y1 = yf + offs;

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

  // First row:
  int ret_val = A * static_cast<int>(l.GetAgastScore(x_left, y_top, 1));
  for (int X = 1; X <= dx; X++) {
    ret_val += r_y_1_i *
        static_cast<int>(l.GetAgastScore(x_left + X, y_top, 1));
  }
  ret_val += B * static_cast<int>(l.GetAgastScore(x_left + dx + 1, y_top, 1));
  // Middle ones:
  for (int Y = 1; Y <= dy; Y++) {
    ret_val += r_x_1_i *
        static_cast<int>(l.GetAgastScore(x_left, y_top + Y, 1));

    for (int X = 1; X <= dx; X++) {
      ret_val += static_cast<int>(l.GetAgastScore(x_left + X, y_top + Y, 1)) *
          scaling;
    }
    ret_val += r_x1_i *
        static_cast<int>(l.GetAgastScore(x_left + dx + 1, y_top + Y, 1));
  }
  // Last row:
  ret_val += D * static_cast<int>(l.GetAgastScore(x_left, y_top + dy + 1, 1));
  for (int X = 1; X <= dx; X++) {
    ret_val += r_y1_i *
        static_cast<int>(l.GetAgastScore(x_left + X, y_top + dy + 1, 1));
  }
  ret_val += C *
      static_cast<int>(l.GetAgastScore(x_left + dx + 1, y_top + dy + 1, 1));

  return ((ret_val + scaling2 / 2) / scaling2);
}

__inline__ bool BriskScaleSpace::IsMax2D(const uint8_t layer, const int x_layer,
                                         const int y_layer) {
  const agast::Mat& scores = pyramid_[layer].scores();
  brisk::BriskLayer& l = pyramid_[layer];
  const int scorescols = scores.cols;
  unsigned char* data = scores.data + y_layer * scorescols + x_layer;
  // Decision tree:
  const unsigned char center = (*data);

  const unsigned char s_10 = l.GetAgastScore(x_layer - 1, y_layer, center);
  if (center < s_10)
    return false;
  const unsigned char s10 = l.GetAgastScore(x_layer + 1, y_layer, center);
  if (center < s10)
    return false;
  const unsigned char s0_1 = l.GetAgastScore(x_layer, y_layer - 1, center);
  if (center < s0_1)
    return false;
  const unsigned char s01 = l.GetAgastScore(x_layer, y_layer + 1, center);
  if (center < s01)
    return false;
  const unsigned char s_11 = l.GetAgastScore(x_layer - 1, y_layer + 1, center);
  if (center < s_11)
    return false;
  const unsigned char s11 = l.GetAgastScore(x_layer + 1, y_layer + 1, center);
  if (center < s11)
    return false;
  const unsigned char s1_1 = l.GetAgastScore(x_layer + 1, y_layer - 1, center);
  if (center < s1_1)
    return false;
  const unsigned char s_1_1 = l.GetAgastScore(x_layer - 1, y_layer - 1, center);
  if (center < s_1_1)
    return false;

  // Reject neighbor maxima.
  std::vector<int> delta;
  // Put together a list of 2d-offsets to where the maximum is also reached.
  if (center == s_1_1) {
    delta.push_back(-1);
    delta.push_back(-1);
  }
  if (center == s0_1) {
    delta.push_back(0);
    delta.push_back(-1);
  }
  if (center == s1_1) {
    delta.push_back(1);
    delta.push_back(-1);
  }
  if (center == s_10) {
    delta.push_back(-1);
    delta.push_back(0);
  }
  if (center == s10) {
    delta.push_back(1);
    delta.push_back(0);
  }
  if (center == s_11) {
    delta.push_back(-1);
    delta.push_back(1);
  }
  if (center == s01) {
    delta.push_back(0);
    delta.push_back(1);
  }
  if (center == s11) {
    delta.push_back(1);
    delta.push_back(1);
  }
  const unsigned int deltasize = delta.size();
  if (deltasize != 0) {
    // In this case, we have to analyze the situation more carefully:
    // the values are gaussian blurred and then we really decide.
    data = scores.data + y_layer * scorescols + x_layer;
    int smoothedcenter = 4 * center + 2 * (s_10 + s10 + s0_1 + s01) + s_1_1
        + s1_1 + s_11 + s11;
    for (unsigned int i = 0; i < deltasize; i += 2) {
      data = scores.data + (y_layer - 1 + delta[i + 1]) * scorescols + x_layer
          + delta[i] - 1;
      int othercenter = *data;
      data++;
      othercenter += 2 * (*data);
      data++;
      othercenter += *data;
      data += scorescols;
      othercenter += 2 * (*data);
      data--;
      othercenter += 4 * (*data);
      data--;
      othercenter += 2 * (*data);
      data += scorescols;
      othercenter += *data;
      data++;
      othercenter += 2 * (*data);
      data++;
      othercenter += *data;
      if (othercenter > smoothedcenter)
        return false;
    }
  }
  return true;
}

// 3D maximum refinement centered around (x_layer,y_layer).
__inline__ float BriskScaleSpace::Refine3D(const uint8_t layer,
                                           const int x_layer, const int y_layer,
                                           float& x, float& y, float& scale,
                                           bool& ismax) {
  ismax = true;
  brisk::BriskLayer& thisLayer = pyramid_[layer];
  const int center = thisLayer.GetAgastScore(x_layer, y_layer, 1);

  // Check and get above maximum:
  float delta_x_above, delta_y_above;
  delta_x_above = delta_y_above = 0;
  float max_above = GetScoreMaxAbove(layer, x_layer, y_layer, center, ismax,
                                     delta_x_above, delta_y_above);

  if (!ismax)
    return 0.0;

  float max;  // To be returned.
  bool doScaleRefinement = true;

  if (layer % 2 == 0) {  // On octave.
    // Treat the patch below:
    float delta_x_below, delta_y_below;
    float max_below_float;
    if (layer == 0) {
      unsigned char max_below_uchar = 0;
      // Guess the lower intra octave...
      BriskLayer& l = pyramid_[0];
      int s_0_0 = l.GetAgastScore_5_8(x_layer - 1, y_layer - 1, 1);
      max_below_uchar = s_0_0;
      int s_1_0 = l.GetAgastScore_5_8(x_layer, y_layer - 1, 1);
      if (s_1_0 > max_below_uchar)
        max_below_uchar = s_1_0;
      int s_2_0 = l.GetAgastScore_5_8(x_layer + 1, y_layer - 1, 1);
      if (s_2_0 > max_below_uchar)
        max_below_uchar = s_2_0;
      int s_2_1 = l.GetAgastScore_5_8(x_layer + 1, y_layer, 1);
      if (s_2_1 > max_below_uchar)
        max_below_uchar = s_2_1;
      int s_1_1 = l.GetAgastScore_5_8(x_layer, y_layer, 1);
      if (s_1_1 > max_below_uchar)
        max_below_uchar = s_1_1;
      int s_0_1 = l.GetAgastScore_5_8(x_layer - 1, y_layer, 1);
      if (s_0_1 > max_below_uchar)
        max_below_uchar = s_0_1;
      int s_0_2 = l.GetAgastScore_5_8(x_layer - 1, y_layer + 1, 1);
      if (s_0_2 > max_below_uchar)
        max_below_uchar = s_0_2;
      int s_1_2 = l.GetAgastScore_5_8(x_layer, y_layer + 1, 1);
      if (s_1_2 > max_below_uchar)
        max_below_uchar = s_1_2;
      int s_2_2 = l.GetAgastScore_5_8(x_layer + 1, y_layer + 1, 1);
      if (s_2_2 > max_below_uchar)
        max_below_uchar = s_2_2;

      max_below_float = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
                                   s_2_0, s_2_1, s_2_2, delta_x_below,
                                   delta_y_below);
      max_below_float = max_below_uchar;
    } else {
      max_below_float = GetScoreMaxBelow(layer, x_layer, y_layer, center, ismax,
                                         delta_x_below, delta_y_below);
      if (!ismax)
        return 0;
    }

    // Get the patch on this layer:
    int s_0_0 = thisLayer.GetAgastScore(x_layer - 1, y_layer - 1, 1);
    int s_1_0 = thisLayer.GetAgastScore(x_layer, y_layer - 1, 1);
    int s_2_0 = thisLayer.GetAgastScore(x_layer + 1, y_layer - 1, 1);
    int s_2_1 = thisLayer.GetAgastScore(x_layer + 1, y_layer, 1);
    int s_1_1 = thisLayer.GetAgastScore(x_layer, y_layer, 1);
    int s_0_1 = thisLayer.GetAgastScore(x_layer - 1, y_layer, 1);
    int s_0_2 = thisLayer.GetAgastScore(x_layer - 1, y_layer + 1, 1);
    int s_1_2 = thisLayer.GetAgastScore(x_layer, y_layer + 1, 1);
    int s_2_2 = thisLayer.GetAgastScore(x_layer + 1, y_layer + 1, 1);

    // Second derivative needs to be sufficiently large.
    if (layer == 0) {
      if (s_1_1 - kMaxThreshold_ <= static_cast<int>(max_above)) {
        doScaleRefinement = false;
      }
    } else {
      if ((s_1_1 - kMaxThreshold_ < (max_above))
          || (s_1_1 - kMaxThreshold_ < (max_below_float))) {
        if ((s_1_1 - kMinDrop_ > (max_above))
            || (s_1_1 - kMinDrop_ > (max_below_float))) {
          // This means, it's an edge on the scale axis.
          doScaleRefinement = false;
        } else {
          // No clear max, no edge -> discard.
          ismax = false;
          return 0.0f;
        }
      }
    }

    float delta_x_layer, delta_y_layer;
    float max_layer = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
                                 s_2_0, s_2_1, s_2_2, delta_x_layer,
                                 delta_y_layer);

    // Calculate the relative scale (1D maximum):
    if (doScaleRefinement) {
      if (layer == 0) {
        scale = Refine1D_2(max_below_float,
                           std::max(static_cast<float>(center), max_layer),
                           max_above, max);
      } else {
        scale = Refine1D(max_below_float,
                         std::max(static_cast<float>(center), max_layer),
                         max_above, max);
      }
    } else {
      scale = 1.0;
      max = max_layer;
    }

    if (scale > 1.0) {
      // Interpolate the position:
      const float r0 = (1.5 - scale) / .5;
      const float r1 = 1.0 - r0;
      x = (r0 * delta_x_layer + r1 * delta_x_above +
          static_cast<float>(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r1 * delta_y_above +
          static_cast<float>(y_layer)) * thisLayer.scale() + thisLayer.offset();
    } else {
      if (layer == 0) {
        // Interpolate the position:
        const float r0 = (scale - 0.5) / 0.5;
        const float r_1 = 1.0 - r0;
        x = r0 * delta_x_layer + r_1 * delta_x_below +
            static_cast<float>(x_layer);
        y = r0 * delta_y_layer + r_1 * delta_y_below +
            static_cast<float>(y_layer);
      } else {
        // Interpolate the position:
        const float r0 = (scale - 0.75) / 0.25;
        const float r_1 = 1.0 - r0;
        x = (r0 * delta_x_layer + r_1 * delta_x_below +
            static_cast<float>(x_layer))
            * thisLayer.scale() + thisLayer.offset();
        y = (r0 * delta_y_layer + r_1 * delta_y_below +
            static_cast<float>(y_layer))
            * thisLayer.scale() + thisLayer.offset();
      }
    }
  } else {
    // On intra.
    // check the patch below:
    float delta_x_below, delta_y_below;
    float max_below = GetScoreMaxBelow(layer, x_layer, y_layer, center, ismax,
                                       delta_x_below, delta_y_below);
    if (!ismax)
      return 0.0;

    // Get the patch on this layer:
    int s_0_0 = thisLayer.GetAgastScore(x_layer - 1, y_layer - 1, 1);
    int s_1_0 = thisLayer.GetAgastScore(x_layer, y_layer - 1, 1);
    int s_2_0 = thisLayer.GetAgastScore(x_layer + 1, y_layer - 1, 1);
    int s_2_1 = thisLayer.GetAgastScore(x_layer + 1, y_layer, 1);
    int s_1_1 = thisLayer.GetAgastScore(x_layer, y_layer, 1);
    int s_0_1 = thisLayer.GetAgastScore(x_layer - 1, y_layer, 1);
    int s_0_2 = thisLayer.GetAgastScore(x_layer - 1, y_layer + 1, 1);
    int s_1_2 = thisLayer.GetAgastScore(x_layer, y_layer + 1, 1);
    int s_2_2 = thisLayer.GetAgastScore(x_layer + 1, y_layer + 1, 1);

    // Second derivative needs to be sufficiently large.
    if ((s_1_1 - kMaxThreshold_ < (max_above))
        || (s_1_1 - kMaxThreshold_ < (max_below))) {
      if ((s_1_1 - kMinDrop_ > (max_above))
          || (s_1_1 - kMinDrop_ > (max_below))) {
        // This means, it's an edge on the scale axis.
        doScaleRefinement = false;
      } else {
        // No clear max, no edge -> discard.
        ismax = false;
        return 0.0f;
      }
    }

    float delta_x_layer, delta_y_layer;
    float max_layer = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
                                 s_2_0, s_2_1, s_2_2, delta_x_layer,
                                 delta_y_layer);

    if (doScaleRefinement) {
      // Calculate the relative scale (1D maximum):
      scale = Refine1D_1(max_below, std::max(static_cast<float>(center),
                                             max_layer),
                         max_above, max);
    } else {
      scale = 1.0;
      max = max_layer;
    }

    if (scale > 1.0) {
      // Interpolate the position:
      const float r0 = 4.0 - scale * 3.0;
      const float r1 = 1.0 - r0;
      x = (r0 * delta_x_layer + r1 * delta_x_above +
          static_cast<float>(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r1 * delta_y_above +
          static_cast<float>(y_layer)) * thisLayer.scale() + thisLayer.offset();
    } else {
      // Interpolate the position:
      const float r0 = scale * 3.0 - 2.0;
      const float r_1 = 1.0 - r0;
      x = (r0 * delta_x_layer + r_1 * delta_x_below +
          static_cast<float>(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r_1 * delta_y_below +
          static_cast<float>(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
  }

  // Calculate the absolute scale:
  scale *= thisLayer.scale();

  // That's it, return the refined maximum:
  return max;
}

// Return the maximum of score patches above or below.
__inline__ float BriskScaleSpace::GetScoreMaxAbove(const uint8_t layer,
                                                   const int x_layer,
                                                   const int y_layer,
                                                   const int thr, bool& ismax,
                                                   float& dx, float& dy) {
  int threshold = thr + kDropThreshold_;

  ismax = false;
  // Relevant floating point coordinates.
  float x_1;
  float x1;
  float y_1;
  float y1;

  // The layer above.
  assert(layer + 1 < layers_);
  brisk::BriskLayer& layerAbove = pyramid_[layer + 1];

  if (layer % 2 == 0) {
    // Octave.
    x_1 = static_cast<float>(4 * (x_layer) - 1 - 2) / 6.0;
    x1 = static_cast<float>(4 * (x_layer) - 1 + 2) / 6.0;
    y_1 = static_cast<float>(4 * (y_layer) - 1 - 2) / 6.0;
    y1 = static_cast<float>(4 * (y_layer) - 1 + 2) / 6.0;
  } else {
    // Intra.
    x_1 = static_cast<float>(6 * (x_layer) - 1 - 3) / 8.0f;
    x1 = static_cast<float>(6 * (x_layer) - 1 + 3) / 8.0f;
    y_1 = static_cast<float>(6 * (y_layer) - 1 - 3) / 8.0f;
    y1 = static_cast<float>(6 * (y_layer) - 1 + 3) / 8.0f;
  }

  // Check the first row.
  int max_x = x_1 + 1;
  int max_y = y_1 + 1;
  float tmp_max;
  float max = layerAbove.GetAgastScore(x_1, y_1, 1);
  if (max > threshold)
    return 0;
  for (int x = x_1 + 1; x <= static_cast<int>(x1); x++) {
    tmp_max = layerAbove.GetAgastScore(static_cast<float>(x), y_1, 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max) {
      max = tmp_max;
      max_x = x;
    }
  }
  tmp_max = layerAbove.GetAgastScore(x1, y_1, 1);
  if (tmp_max > threshold)
    return 0;
  if (tmp_max > max) {
    max = tmp_max;
    max_x = static_cast<int>(x1);
  }

  // Middle rows.
  for (int y = y_1 + 1; y <= static_cast<int>(y1); y++) {
    tmp_max = layerAbove.GetAgastScore(x_1, static_cast<float>(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max) {
      max = tmp_max;
      max_x = static_cast<int>(x_1 + 1);
      max_y = y;
    }
    for (int x = x_1 + 1; x <= static_cast<int>(x1); x++) {
      tmp_max = layerAbove.GetAgastScore(x, y, 1);
      if (tmp_max > threshold)
        return 0;
      if (tmp_max > max) {
        max = tmp_max;
        max_x = x;
        max_y = y;
      }
    }
    tmp_max = layerAbove.GetAgastScore(x1, static_cast<float>(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max) {
      max = tmp_max;
      max_x = static_cast<int>(x1);
      max_y = y;
    }
  }

  // Bottom row.
  tmp_max = layerAbove.GetAgastScore(x_1, y1, 1);
  if (tmp_max > max) {
    max = tmp_max;
    max_x = static_cast<int>(x_1 + 1);
    max_y = static_cast<int>(y1);
  }
  for (int x = x_1 + 1; x <= static_cast<int>(x1); x++) {
    tmp_max = layerAbove.GetAgastScore(static_cast<float>(x), y1, 1);
    if (tmp_max > max) {
      max = tmp_max;
      max_x = x;
      max_y = static_cast<int>(y1);
    }
  }
  tmp_max = layerAbove.GetAgastScore(x1, y1, 1);
  if (tmp_max > max) {
    max = tmp_max;
    max_x = static_cast<int>(x1);
    max_y = static_cast<int>(y1);
  }

  // Find dx / dy:
  int s_0_0 = layerAbove.GetAgastScore(max_x - 1, max_y - 1, 1);
  int s_1_0 = layerAbove.GetAgastScore(max_x, max_y - 1, 1);
  int s_2_0 = layerAbove.GetAgastScore(max_x + 1, max_y - 1, 1);
  int s_2_1 = layerAbove.GetAgastScore(max_x + 1, max_y, 1);
  int s_1_1 = layerAbove.GetAgastScore(max_x, max_y, 1);
  int s_0_1 = layerAbove.GetAgastScore(max_x - 1, max_y, 1);
  int s_0_2 = layerAbove.GetAgastScore(max_x - 1, max_y + 1, 1);
  int s_1_2 = layerAbove.GetAgastScore(max_x, max_y + 1, 1);
  int s_2_2 = layerAbove.GetAgastScore(max_x + 1, max_y + 1, 1);
  float dx_1, dy_1;
  float refined_max = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
                                 s_2_0, s_2_1, s_2_2, dx_1, dy_1);

  // Calculate dx / dy in above coordinates.
  float real_x = static_cast<float>(max_x) + dx_1;
  float real_y = static_cast<float>(max_y) + dy_1;
  bool returnrefined = true;
  if (layer % 2 == 0) {
    dx = (real_x * 6.0f + 1.0f) / 4.0f - static_cast<float>(x_layer);
    dy = (real_y * 6.0f + 1.0f) / 4.0f - static_cast<float>(y_layer);
  } else {
    dx = (real_x * 8.0 + 1.0) / 6.0 - static_cast<float>(x_layer);
    dy = (real_y * 8.0 + 1.0) / 6.0 - static_cast<float>(y_layer);
  }

  // Saturate.
  if (dx > 1.0f) {
    dx = 1.0f;
    returnrefined = false;
  }
  if (dx < -1.0f) {
    dx = -1.0f;
    returnrefined = false;
  }
  if (dy > 1.0f) {
    dy = 1.0f;
    returnrefined = false;
  }
  if (dy < -1.0f) {
    dy = -1.0f;
    returnrefined = false;
  }

  // Done and ok.
  ismax = true;
  if (returnrefined) {
    return std::max(refined_max, max);
  }
  return max;
}

__inline__ float BriskScaleSpace::GetScoreMaxBelow(const uint8_t layer,
                                                   const int x_layer,
                                                   const int y_layer,
                                                   const int thr, bool& ismax,
                                                   float& dx, float& dy) {
  int threshold = thr + kDropThreshold_;

  ismax = false;

  // Relevant floating point coordinates.
  float x_1;
  float x1;
  float y_1;
  float y1;

  if (layer % 2 == 0) {
    // Octave.
    x_1 = static_cast<float>(8 * (x_layer) + 1 - 4) / 6.0;
    x1 = static_cast<float>(8 * (x_layer) + 1 + 4) / 6.0;
    y_1 = static_cast<float>(8 * (y_layer) + 1 - 4) / 6.0;
    y1 = static_cast<float>(8 * (y_layer) + 1 + 4) / 6.0;
  } else {
    x_1 = static_cast<float>(6 * (x_layer) + 1 - 3) / 4.0;
    x1 = static_cast<float>(6 * (x_layer) + 1 + 3) / 4.0;
    y_1 = static_cast<float>(6 * (y_layer) + 1 - 3) / 4.0;
    y1 = static_cast<float>(6 * (y_layer) + 1 + 3) / 4.0;
  }

  // The layer below.
  assert(layer > 0);
  brisk::BriskLayer& layerBelow = pyramid_[layer - 1];

  // Check the first row.
  int max_x = x_1 + 1;
  int max_y = y_1 + 1;
  float tmp_max;
  float max = layerBelow.GetAgastScore(x_1, y_1, 1);
  if (max > threshold)
    return 0;
  for (int x = x_1 + 1; x <= static_cast<int>(x1); x++) {
    tmp_max = layerBelow.GetAgastScore(static_cast<float>(x), y_1, 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max) {
      max = tmp_max;
      max_x = x;
    }
  }
  tmp_max = layerBelow.GetAgastScore(x1, y_1, 1);
  if (tmp_max > threshold)
    return 0;
  if (tmp_max > max) {
    max = tmp_max;
    max_x = static_cast<int>(x1);
  }

  // Middle rows.
  for (int y = y_1 + 1; y <= static_cast<int>(y1); y++) {
    tmp_max = layerBelow.GetAgastScore(x_1, static_cast<float>(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max) {
      max = tmp_max;
      max_x = static_cast<int>(x_1 + 1);
      max_y = y;
    }
    for (int x = x_1 + 1; x <= static_cast<int>(x1); x++) {
      tmp_max = layerBelow.GetAgastScore(x, y, 1);
      if (tmp_max > threshold)
        return 0;
      if (tmp_max == max) {
        const int t1 = 2
            * (layerBelow.GetAgastScore(x - 1, y, 1)
                + layerBelow.GetAgastScore(x + 1, y, 1)
                + layerBelow.GetAgastScore(x, y + 1, 1)
                + layerBelow.GetAgastScore(x, y - 1, 1))
            + (layerBelow.GetAgastScore(x + 1, y + 1, 1)
                + layerBelow.GetAgastScore(x - 1, y + 1, 1)
                + layerBelow.GetAgastScore(x + 1, y - 1, 1)
                + layerBelow.GetAgastScore(x - 1, y - 1, 1));
        const int t2 = 2
            * (layerBelow.GetAgastScore(max_x - 1, max_y, 1)
                + layerBelow.GetAgastScore(max_x + 1, max_y, 1)
                + layerBelow.GetAgastScore(max_x, max_y + 1, 1)
                + layerBelow.GetAgastScore(max_x, max_y - 1, 1))
            + (layerBelow.GetAgastScore(max_x + 1, max_y + 1, 1)
                + layerBelow.GetAgastScore(max_x - 1, max_y + 1, 1)
                + layerBelow.GetAgastScore(max_x + 1, max_y - 1, 1)
                + layerBelow.GetAgastScore(max_x - 1, max_y - 1, 1));
        if (t1 > t2) {
          max_x = x;
          max_y = y;
        }
      }
      if (tmp_max > max) {
        max = tmp_max;
        max_x = x;
        max_y = y;
      }
    }
    tmp_max = layerBelow.GetAgastScore(x1, static_cast<float>(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max) {
      max = tmp_max;
      max_x = static_cast<int>(x1);
      max_y = y;
    }
  }

  // Bottom row.
  tmp_max = layerBelow.GetAgastScore(x_1, y1, 1);
  if (tmp_max > max) {
    max = tmp_max;
    max_x = static_cast<int>(x_1 + 1);
    max_y = static_cast<int>(y1);
  }
  for (int x = x_1 + 1; x <= static_cast<int>(x1); x++) {
    tmp_max = layerBelow.GetAgastScore(static_cast<float>(x), y1, 1);
    if (tmp_max > max) {
      max = tmp_max;
      max_x = x;
      max_y = static_cast<int>(y1);
    }
  }
  tmp_max = layerBelow.GetAgastScore(x1, y1, 1);
  if (tmp_max > max) {
    max = tmp_max;
    max_x = static_cast<int>(x1);
    max_y = static_cast<int>(y1);
  }

  // Find dx/dy:
  int s_0_0 = layerBelow.GetAgastScore(max_x - 1, max_y - 1, 1);
  int s_1_0 = layerBelow.GetAgastScore(max_x, max_y - 1, 1);
  int s_2_0 = layerBelow.GetAgastScore(max_x + 1, max_y - 1, 1);
  int s_2_1 = layerBelow.GetAgastScore(max_x + 1, max_y, 1);
  int s_1_1 = layerBelow.GetAgastScore(max_x, max_y, 1);
  int s_0_1 = layerBelow.GetAgastScore(max_x - 1, max_y, 1);
  int s_0_2 = layerBelow.GetAgastScore(max_x - 1, max_y + 1, 1);
  int s_1_2 = layerBelow.GetAgastScore(max_x, max_y + 1, 1);
  int s_2_2 = layerBelow.GetAgastScore(max_x + 1, max_y + 1, 1);
  float dx_1, dy_1;
  float refined_max = Subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
                                 s_2_0, s_2_1, s_2_2, dx_1, dy_1);

  // Calculate dx/dy in above coordinates.
  float real_x = static_cast<float>(max_x) + dx_1;
  float real_y = static_cast<float>(max_y) + dy_1;
  bool returnrefined = true;
  if (layer % 2 == 0) {
    dx = (real_x * 6.0 + 1.0) / 8.0 - static_cast<float>(x_layer);
    dy = (real_y * 6.0 + 1.0) / 8.0 - static_cast<float>(y_layer);
  } else {
    dx = (real_x * 4.0 - 1.0) / 6.0 - static_cast<float>(x_layer);
    dy = (real_y * 4.0 - 1.0) / 6.0 - static_cast<float>(y_layer);
  }

  // Saturate.
  if (dx > 1.0) {
    dx = 1.0;
    returnrefined = false;
  }
  if (dx < -1.0) {
    dx = -1.0;
    returnrefined = false;
  }
  if (dy > 1.0) {
    dy = 1.0;
    returnrefined = false;
  }
  if (dy < -1.0) {
    dy = -1.0;
    returnrefined = false;
  }

  // Done and ok.
  ismax = true;
  if (returnrefined) {
    return std::max(refined_max, max);
  }
  return max;
}

__inline__ float BriskScaleSpace::Refine1D(const float s_05, const float s0,
                                           const float s05, float& max) {
  int i_05 = static_cast<int>(1024.0 * s_05 + 0.5);
  int i0 = static_cast<int>(1024.0 * s0 + 0.5);
  int i05 = static_cast<int>(1024.0 * s05 + 0.5);

  //   16.0000  -24.0000    8.0000
  //  -40.0000   54.0000  -14.0000
  //   24.0000  -27.0000    6.0000

  int three_a = 16 * i_05 - 24 * i0 + 8 * i05;
  // Second derivative must be negative:
  if (three_a >= 0) {
    if (s0 >= s_05 && s0 >= s05) {
      max = s0;
      return 1.0;
    }
    if (s_05 >= s0 && s_05 >= s05) {
      max = s_05;
      return 0.75;
    }
    if (s05 >= s0 && s05 >= s_05) {
      max = s05;
      return 1.5;
    }
  }

  int three_b = -40 * i_05 + 54 * i0 - 14 * i05;
  // Calculate max location:
  float ret_val = -static_cast<float>(three_b) /
      static_cast<float>(2 * three_a);
  // Saturate and return.
  if (ret_val < 0.75)
    ret_val = 0.75;
  else if (ret_val > 1.5)
    ret_val = 1.5;  // Allow to be slightly off bounds ...?
  int three_c = +24 * i_05 - 27 * i0 + 6 * i05;
  max = static_cast<float>(three_c) +
      static_cast<float>(three_a) * ret_val * ret_val
      + static_cast<float>(three_b) * ret_val;
  max /= 3072.0;
  return ret_val;
}

__inline__ float BriskScaleSpace::Refine1D_1(const float s_05, const float s0,
                                             const float s05, float& max) {
  int i_05 = static_cast<int>(1024.0 * s_05 + 0.5);
  int i0 = static_cast<int>(1024.0 * s0 + 0.5);
  int i05 = static_cast<int>(1024.0 * s05 + 0.5);

  //  4.5000   -9.0000    4.5000
  // -10.5000   18.0000   -7.5000
  //  6.0000   -8.0000    3.0000

  int two_a = 9 * i_05 - 18 * i0 + 9 * i05;
  // Second derivative must be negative:
  if (two_a >= 0) {
    if (s0 >= s_05 && s0 >= s05) {
      max = s0;
      return 1.0;
    }
    if (s_05 >= s0 && s_05 >= s05) {
      max = s_05;
      return 0.6666666666666666666666666667;
    }
    if (s05 >= s0 && s05 >= s_05) {
      max = s05;
      return 1.3333333333333333333333333333;
    }
  }

  int two_b = -21 * i_05 + 36 * i0 - 15 * i05;
  // Calculate max location:
  float ret_val = -static_cast<float>(two_b) / static_cast<float>(2 * two_a);
  // Saturate and return.
  if (ret_val < 0.6666666666666666666666666667)
    ret_val = 0.666666666666666666666666667;
  else if (ret_val > 1.33333333333333333333333333)
    ret_val = 1.333333333333333333333333333;
  int two_c = +12 * i_05 - 16 * i0 + 6 * i05;
  max = static_cast<float>(two_c) +
      static_cast<float>(two_a) * ret_val * ret_val +
      static_cast<float>(two_b) * ret_val;
  max /= 2048.0;
  return ret_val;
}

__inline__ float BriskScaleSpace::Refine1D_2(const float s_05, const float s0,
                                             const float s05, float& max) {
  int i_05 = static_cast<int>(1024.0 * s_05 + 0.5);
  int i0 = static_cast<int>(1024.0 * s0 + 0.5);
  int i05 = static_cast<int>(1024.0 * s05 + 0.5);

  //   18.0000  -30.0000   12.0000
  //  -45.0000   65.0000  -20.0000
  //   27.0000  -30.0000    8.0000

  int a = 2 * i_05 - 4 * i0 + 2 * i05;
  // Second derivative must be negative:
  if (a >= 0) {
    if (s0 >= s_05 && s0 >= s05) {
      max = s0;
      return 1.0;
    }
    if (s_05 >= s0 && s_05 >= s05) {
      max = s_05;
      return 0.7;
    }
    if (s05 >= s0 && s05 >= s_05) {
      max = s05;
      return 1.5;
    }
  }

  int b = -5 * i_05 + 8 * i0 - 3 * i05;
  // Calculate max location:
  float ret_val = -static_cast<float>(b) / static_cast<float>(2 * a);
  // Saturate and return.
  if (ret_val < 0.7)
    ret_val = 0.7;
  else if (ret_val > 1.5)
    ret_val = 1.5;  // Allow to be slightly off bounds ...?
  int c = +3 * i_05 - 3 * i0 + 1 * i05;
  max = static_cast<float>(c) + static_cast<float>(a) * ret_val * ret_val +
      static_cast<float>(b) * ret_val;
  max /= 1024;
  return ret_val;
}

__inline__ float BriskScaleSpace::Subpixel2D(const int s_0_0, const int s_0_1,
                                             const int s_0_2, const int s_1_0,
                                             const int s_1_1, const int s_1_2,
                                             const int s_2_0, const int s_2_1,
                                             const int s_2_2, float& delta_x,
                                             float& delta_y) {
  // The coefficients of the 2d quadratic function least-squares fit:
  int tmp1 = s_0_0 + s_0_2 - 2 * s_1_1 + s_2_0 + s_2_2;
  int coeff1 = 3 * (tmp1 + s_0_1 - ((s_1_0 + s_1_2) << 1) + s_2_1);
  int coeff2 = 3 * (tmp1 - ((s_0_1 + s_2_1) << 1) + s_1_0 + s_1_2);
  int tmp2 = s_0_2 - s_2_0;
  int tmp3 = (s_0_0 + tmp2 - s_2_2);
  int tmp4 = tmp3 - 2 * tmp2;
  int coeff3 = -3 * (tmp3 + s_0_1 - s_2_1);
  int coeff4 = -3 * (tmp4 + s_1_0 - s_1_2);
  int coeff5 = (s_0_0 - s_0_2 - s_2_0 + s_2_2) << 2;
  int coeff6 = -(s_0_0 + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1) << 1)
      - 5 * s_1_1 + s_2_0 + s_2_2) << 1;

  // 2nd derivative test:
  int H_det = 4 * coeff1 * coeff2 - coeff5 * coeff5;

  if (H_det == 0) {
    delta_x = 0.0;
    delta_y = 0.0;
    return static_cast<float>(coeff6) / 18.0;
  }

  if (!(H_det > 0 && coeff1 < 0)) {
    // The maximum must be at the one of the 4 patch corners.
    int tmp_max = coeff3 + coeff4 + coeff5;
    delta_x = 1.0;
    delta_y = 1.0;

    int tmp = -coeff3 + coeff4 - coeff5;
    if (tmp > tmp_max) {
      tmp_max = tmp;
      delta_x = -1.0;
      delta_y = 1.0;
    }
    tmp = coeff3 - coeff4 - coeff5;
    if (tmp > tmp_max) {
      tmp_max = tmp;
      delta_x = 1.0;
      delta_y = -1.0;
    }
    tmp = -coeff3 - coeff4 + coeff5;
    if (tmp > tmp_max) {
      tmp_max = tmp;
      delta_x = -1.0;
      delta_y = -1.0;
    }
    return static_cast<float>(tmp_max + coeff1 + coeff2 + coeff6) / 18.0;
  }

  // This is hopefully the normal outcome of the Hessian test.
  delta_x = static_cast<float>(2 * coeff2 * coeff3 - coeff4 * coeff5) /
      static_cast<float>(-H_det);
  delta_y = static_cast<float>(2 * coeff1 * coeff4 - coeff3 * coeff5) /
      static_cast<float>(-H_det);
  // TODO(lestefan): this is not correct, but easy, so perform a real boundary
  // maximum search:
  bool tx = false;
  bool tx_ = false;
  bool ty = false;
  bool ty_ = false;
  if (delta_x > 1.0)
    tx = true;
  else if (delta_x < -1.0)
    tx_ = true;
  if (delta_y > 1.0)
    ty = true;
  if (delta_y < -1.0)
    ty_ = true;

  if (tx || tx_ || ty || ty_) {
    // Get two candidates:
    float delta_x1 = 0.0, delta_x2 = 0.0, delta_y1 = 0.0, delta_y2 = 0.0;
    if (tx) {
      delta_x1 = 1.0;
      delta_y1 = -static_cast<float>(coeff4 + coeff5) /
          static_cast<float>(2 * coeff2);
      if (delta_y1 > 1.0)
        delta_y1 = 1.0;
      else if (delta_y1 < -1.0)
        delta_y1 = -1.0;
    } else if (tx_) {
      delta_x1 = -1.0;
      delta_y1 = -static_cast<float>(coeff4 - coeff5) /
          static_cast<float>(2 * coeff2);
      if (delta_y1 > 1.0)
        delta_y1 = 1.0;
      else if (delta_y1 < -1.0)
        delta_y1 = -1.0;
    }
    if (ty) {
      delta_y2 = 1.0;
      delta_x2 = -static_cast<float>(coeff3 + coeff5) /
          static_cast<float>(2 * coeff1);
      if (delta_x2 > 1.0)
        delta_x2 = 1.0;
      else if (delta_x2 < -1.0)
        delta_x2 = -1.0;
    } else if (ty_) {
      delta_y2 = -1.0;
      delta_x2 = -static_cast<float>(coeff3 - coeff5) /
          static_cast<float>(2 * coeff1);
      if (delta_x2 > 1.0)
        delta_x2 = 1.0;
      else if (delta_x2 < -1.0)
        delta_x2 = -1.0;
    }
    // Insert both options for evaluation which to pick.
    float max1 = (coeff1 * delta_x1 * delta_x1 + coeff2 * delta_y1 * delta_y1
        + coeff3 * delta_x1 + coeff4 * delta_y1 + coeff5 * delta_x1 * delta_y1
        + coeff6) / 18.0;
    float max2 = (coeff1 * delta_x2 * delta_x2 + coeff2 * delta_y2 * delta_y2
        + coeff3 * delta_x2 + coeff4 * delta_y2 + coeff5 * delta_x2 * delta_y2
        + coeff6) / 18.0;
    if (max1 > max2) {
      delta_x = delta_x1;
      delta_y = delta_x1;
      return max1;
    } else {
      delta_x = delta_x2;
      delta_y = delta_x2;
      return max2;
    }
  }

  // This is the case of the maximum inside the boundaries:
  return (coeff1 * delta_x * delta_x + coeff2 * delta_y * delta_y
      + coeff3 * delta_x + coeff4 * delta_y + coeff5 * delta_x * delta_y
      + coeff6) / 18.0;
}

}  // namespace brisk
