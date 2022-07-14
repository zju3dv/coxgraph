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

#include <bitset>
#include <istream>  // NOLINT
#include <fstream>  // NOLINT
#include <iostream>  // NOLINT

#include <stdexcept>

#include <brisk/brisk-descriptor-extractor.h>
#include <agast/wrap-opencv.h>
#include <brisk/internal/helper-structures.h>
#include <brisk/internal/integral-image.h>
#include <brisk/internal/macros.h>
#include <brisk/internal/pattern-provider.h>
#include <brisk/internal/timer.h>

namespace brisk {
const float BriskDescriptorExtractor::basicSize_ = 12.0;
const unsigned int BriskDescriptorExtractor::scales_ = 64;
// 40->4 Octaves - else, this needs to be adjusted...
const float BriskDescriptorExtractor::scalerange_ = 30;
// Discretization of the rotation look-up.
const unsigned int BriskDescriptorExtractor::n_rot_ = 1024;

// legacy BRISK 1.0
void BriskDescriptorExtractor::generateKernel(std::vector<float> &radiusList,
                                              std::vector<int> &numberList,
                                              float dMax, float dMin,
                                              std::vector<int> indexChange) {

  dMax_ = dMax;
  dMin_ = dMin;

  // get the total number of points
  const int rings = radiusList.size();
  assert(radiusList.size() != 0 && radiusList.size() == numberList.size());
  points_ = 0;  // remember the total number of points
  for (int ring = 0; ring < rings; ring++) {
    points_ += numberList[ring];
  }
  // set up the patterns
  patternPoints_ = new BriskPatternPoint[points_ * scales_ * n_rot_];
  BriskPatternPoint* patternIterator = patternPoints_;

  // define the scale discretization:
  static const float lb_scale = log(scalerange_) / log(2.0);
  static const float lb_scale_step = lb_scale / (scales_);

  scaleList_ = new float[scales_];
  sizeList_ = new unsigned int[scales_];

  const float sigma_scale = 1.3;

  for (unsigned int scale = 0; scale < scales_; ++scale) {
    scaleList_[scale] = pow((double) 2.0, (double) (scale * lb_scale_step));
    sizeList_[scale] = 0;

    // generate the pattern points look-up
    double alpha, theta;
    for (size_t rot = 0; rot < n_rot_; ++rot) {
      theta = double(rot) * 2 * M_PI / double(n_rot_);  // this is the rotation of the feature
      for (int ring = 0; ring < rings; ++ring) {
        for (int num = 0; num < numberList[ring]; ++num) {
          // the actual coordinates on the circle
          alpha = (double(num)) * 2 * M_PI / double(numberList[ring]);
          patternIterator->x = scaleList_[scale] * radiusList[ring]
              * cos(alpha + theta);  // feature rotation plus angle of the point
          patternIterator->y = scaleList_[scale] * radiusList[ring]
              * sin(alpha + theta);
          // and the gaussian kernel sigma
          if (ring == 0) {
            patternIterator->sigma = sigma_scale * scaleList_[scale] * 0.5;
          } else {
            patternIterator->sigma = sigma_scale * scaleList_[scale]
                * (double(radiusList[ring])) * sin(M_PI / numberList[ring]);
          }

          // adapt the sizeList if necessary
          const unsigned int size = ceil(
              ((scaleList_[scale] * radiusList[ring]) + patternIterator->sigma))
              + 1;
          if (sizeList_[scale] < size) {
            sizeList_[scale] = size;
          }

          // increment the iterator
          ++patternIterator;
        }
      }
    }
  }

  // now also generate pairings
  shortPairs_ = new BriskShortPair[points_ * (points_ - 1) / 2];
  longPairs_ = new BriskLongPair[points_ * (points_ - 1) / 2];
  noShortPairs_ = 0;
  noLongPairs_ = 0;

  // fill indexChange with 0..n if empty
  unsigned int indSize = indexChange.size();
  if (indSize == 0) {
    indexChange.resize(points_ * (points_ - 1) / 2);
    indSize = indexChange.size();
    for (unsigned int i = 0; i < indSize; i++) {
      indexChange[i] = i;
    }
  }
  const float dMin_sq = dMin_ * dMin_;
  const float dMax_sq = dMax_ * dMax_;
  for (unsigned int i = 1; i < points_; i++) {
    for (unsigned int j = 0; j < i; j++) {  //(find all the pairs)
      // point pair distance:
      const float dx = patternPoints_[j].x - patternPoints_[i].x;
      const float dy = patternPoints_[j].y - patternPoints_[i].y;
      const float norm_sq = (dx * dx + dy * dy);
      if (norm_sq > dMin_sq) {
        // save to long pairs
        BriskLongPair& longPair = longPairs_[noLongPairs_];
        longPair.weighted_dx = int((dx / (norm_sq)) * 2048.0 + 0.5);
        longPair.weighted_dy = int((dy / (norm_sq)) * 2048.0 + 0.5);
        longPair.i = i;
        longPair.j = j;
        ++noLongPairs_;
      }
      if (norm_sq < dMax_sq) {
        // save to short pairs
        assert(noShortPairs_ < indSize);  // make sure the user passes something sensible
        BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
        shortPair.j = j;
        shortPair.i = i;
        ++noShortPairs_;
      }
    }
  }

// no bits:
strings_=(int)ceil((float(noShortPairs_))/128.0)*4*4;

}

void BriskDescriptorExtractor::InitFromStream(bool rotationInvariant,
                                              bool scaleInvariant,
                                              std::istream& pattern_stream,
                                              float patternScale) {
  // Not in use.
  dMax_ = 0;
  dMin_ = 0;
  rotationInvariance = rotationInvariant;
  scaleInvariance = scaleInvariant;

  assert(pattern_stream.good());

  // Read number of points.
  pattern_stream >> points_;

  // Set up the patterns.
  patternPoints_ = new brisk::BriskPatternPoint[points_ * scales_ * n_rot_];
  brisk::BriskPatternPoint* patternIterator = patternPoints_;

  // Define the scale discretization:
  static const float lb_scale = log(scalerange_) / log(2.0);
  static const float lb_scale_step = lb_scale / (scales_);

  scaleList_ = new float[scales_];
  sizeList_ = new unsigned int[scales_];

  const float sigma_scale = 1.3;

  // First fill the unscaled and unrotated pattern:
  float* u_x = new float[points_];
  float* u_y = new float[points_];
  float* sigma = new float[points_];
  for (unsigned int i = 0; i < points_; i++) {
    pattern_stream >> u_x[i]; u_x[i]*=patternScale;
    pattern_stream >> u_y[i]; u_y[i]*=patternScale;
    pattern_stream >> sigma[i]; sigma[i]*=patternScale;
  }

  // Now fill all the scaled and rotated versions.
  for (unsigned int scale = 0; scale < scales_; ++scale) {
    scaleList_[scale] = pow(2.0, static_cast<double>(scale * lb_scale_step));
    sizeList_[scale] = 0;

    // Generate the pattern points look-up.
    double theta;
    for (size_t rot = 0; rot < n_rot_; ++rot) {
      for (unsigned int i = 0; i < points_; i++) {
        // This is the rotation of the feature.
        theta = static_cast<double>(rot) * 2 * M_PI
            / static_cast<double>(n_rot_);
        // Feature rotation plus angle of the point.
        patternIterator->x = scaleList_[scale]
            * (u_x[i] * cos(theta) - u_y[i] * sin(theta));
        patternIterator->y = scaleList_[scale]
            * (u_x[i] * sin(theta) + u_y[i] * cos(theta));
        // And the Gaussian kernel sigma.
        patternIterator->sigma = sigma_scale * scaleList_[scale] * sigma[i];

        // Adapt the sizeList if necessary.
        const unsigned int size = ceil(
            ((sqrt(
                patternIterator->x * patternIterator->x
                    + patternIterator->y * patternIterator->y))
                + patternIterator->sigma)) + 1;
        if (sizeList_[scale] < size) {
          sizeList_[scale] = size;
        }

        // Increment the iterator.
        ++patternIterator;
      }
    }
  }

  // Now also generate pairings.
  pattern_stream >> noShortPairs_;
  shortPairs_ = new brisk::BriskShortPair[noShortPairs_];
  for (unsigned int p = 0; p < noShortPairs_; p++) {
    unsigned int i, j;
    pattern_stream >> i;
    shortPairs_[p].i = i;
    pattern_stream >> j;
    shortPairs_[p].j = j;
  }

  pattern_stream >> noLongPairs_;
  longPairs_ = new brisk::BriskLongPair[noLongPairs_];
  for (unsigned int p = 0; p < noLongPairs_; p++) {
    unsigned int i, j;
    pattern_stream >> i;
    longPairs_[p].i = i;
    pattern_stream >> j;
    longPairs_[p].j = j;
    float dx = (u_x[j] - u_x[i]);
    float dy = (u_y[j] - u_y[i]);
    float norm_sq = dx * dx + dy * dy;
    longPairs_[p].weighted_dx =
        static_cast<int>((dx / (norm_sq)) * 2048.0 + 0.5);
    longPairs_[p].weighted_dy =
        static_cast<int>((dy / (norm_sq)) * 2048.0 + 0.5);
  }

  // Number of descriptor bits:
  strings_ = static_cast<int>(ceil((static_cast<float>(noShortPairs_)) / 128.0))
      * 4 * 4;

  CHECK_EQ(noShortPairs_, kDescriptorLength);

  delete[] u_x;
  delete[] u_y;
  delete[] sigma;
}

BriskDescriptorExtractor::BriskDescriptorExtractor() :
  BriskDescriptorExtractor(true, true) { }

BriskDescriptorExtractor::BriskDescriptorExtractor(bool rotationInvariant,
                                                   bool scaleInvariant) :
  BriskDescriptorExtractor(rotationInvariant, scaleInvariant,
                           Version::briskV2, 1.0) { }

BriskDescriptorExtractor::BriskDescriptorExtractor(bool rotationInvariant,
                                                   bool scaleInvariant,
                                                   int version) :
  BriskDescriptorExtractor(rotationInvariant, scaleInvariant,
                           version, 1.0) { }

BriskDescriptorExtractor::BriskDescriptorExtractor(bool rotationInvariant,
                                                   bool scaleInvariant,
                                                   int version,
                                                   float patternScale) {
  CHECK(version == Version::briskV1 || version == Version::briskV2);
  if(version == Version::briskV2){
    std::stringstream ss;
    brisk::GetDefaultPatternAsStream(&ss);
    InitFromStream(rotationInvariant, scaleInvariant, ss, patternScale);
  } else if(version == Version::briskV1){
    std::vector<float> rList;
    std::vector<int> nList;

    // this is the standard pattern found to be suitable also
    rList.resize(5);
    nList.resize(5);
    const double f = 0.85 * patternScale;

    rList[0] = f * 0;
    rList[1] = f * 2.9;
    rList[2] = f * 4.9;
    rList[3] = f * 7.4;
    rList[4] = f * 10.8;

    nList[0] = 1;
    nList[1] = 10;
    nList[2] = 14;
    nList[3] = 15;
    nList[4] = 20;

    rotationInvariance = rotationInvariant;
    scaleInvariance = scaleInvariant;
    generateKernel(rList, nList, 5.85 , 8.2 );
  } else {
    throw std::runtime_error("only Version::briskV1 or Version::briskV2 supported!");
  }
}

BriskDescriptorExtractor::BriskDescriptorExtractor(const std::string& fname) :
  BriskDescriptorExtractor(fname, true) { }

BriskDescriptorExtractor::BriskDescriptorExtractor(const std::string& fname,
                                                   bool rotationInvariant) :
  BriskDescriptorExtractor(fname, rotationInvariant, true) { }

BriskDescriptorExtractor::BriskDescriptorExtractor(const std::string& fname,
                                                   bool rotationInvariant,
                                                   bool scaleInvariant) :
  BriskDescriptorExtractor(fname, rotationInvariant, scaleInvariant, 1.0) { }

BriskDescriptorExtractor::BriskDescriptorExtractor(const std::string& fname,
                                                   bool rotationInvariant,
                                                   bool scaleInvariant,
                                                   float patternScale) {
  std::ifstream myfile(fname.c_str());
  assert(myfile.is_open());

  InitFromStream(rotationInvariant, scaleInvariant, myfile, patternScale);

  myfile.close();
}

// Simple alternative:
template<typename ImgPixel_T, typename IntegralPixel_T>
__inline__ IntegralPixel_T BriskDescriptorExtractor::SmoothedIntensity(
    const agast::Mat& image, const agast::Mat& integral, const float key_x,
    const float key_y, const unsigned int scale, const unsigned int rot,
    const unsigned int point) const {
  // Get the float position.
  const brisk::BriskPatternPoint& briskPoint = patternPoints_[scale * n_rot_
      * points_ + rot * points_ + point];

  const float xf = briskPoint.x + key_x;
  const float yf = briskPoint.y + key_y;
  const int x = static_cast<int>(xf);
  const int y = static_cast<int>(yf);
  const int& imagecols = image.cols;

  // Get the sigma:
  const float sigma_half = briskPoint.sigma;
  const float area = 4.0 * sigma_half * sigma_half;

  // Calculate output:
  int ret_val;
  if (sigma_half < 0.5) {
    // Interpolation multipliers:
    const int r_x = (xf - x) * 1024;
    const int r_y = (yf - y) * 1024;
    const int r_x_1 = (1024 - r_x);
    const int r_y_1 = (1024 - r_y);
    ImgPixel_T* ptr = reinterpret_cast<ImgPixel_T*>(image.data) + x
        + y * imagecols;
    // Just interpolate:
    ret_val = (r_x_1 * r_y_1 * IntegralPixel_T(*ptr));
    ptr++;
    ret_val += (r_x * r_y_1 * IntegralPixel_T(*ptr));
    ptr += imagecols;
    ret_val += (r_x * r_y * IntegralPixel_T(*ptr));
    ptr--;
    ret_val += (r_x_1 * r_y * IntegralPixel_T(*ptr));
    return (ret_val) / 1024;
  }

  // This is the standard case (simple, not speed optimized yet):
  // Scaling:
  const IntegralPixel_T scaling = 4194304.0 / area;
  const IntegralPixel_T scaling2 = static_cast<float>(scaling) * area / 1024.0;

  // The integral image is larger:
  const int integralcols = imagecols + 1;

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
  const IntegralPixel_T A = (r_x_1 * r_y_1) * scaling;
  const IntegralPixel_T B = (r_x1 * r_y_1) * scaling;
  const IntegralPixel_T C = (r_x1 * r_y1) * scaling;
  const IntegralPixel_T D = (r_x_1 * r_y1) * scaling;
  const IntegralPixel_T r_x_1_i = r_x_1 * scaling;
  const IntegralPixel_T r_y_1_i = r_y_1 * scaling;
  const IntegralPixel_T r_x1_i = r_x1 * scaling;
  const IntegralPixel_T r_y1_i = r_y1 * scaling;

  if (dx + dy > 2) {
    // Now the calculation:
    ImgPixel_T* ptr = reinterpret_cast<ImgPixel_T*>(image.data) + x_left
        + imagecols * y_top;
    // First the corners:
    ret_val = A * IntegralPixel_T(*ptr);
    ptr += dx + 1;
    ret_val += B * IntegralPixel_T(*ptr);
    ptr += dy * imagecols + 1;
    ret_val += C * IntegralPixel_T(*ptr);
    ptr -= dx + 1;
    ret_val += D * IntegralPixel_T(*ptr);

    // Next the edges:
    IntegralPixel_T* ptr_integral = reinterpret_cast<IntegralPixel_T*>(integral
        .data) + x_left + integralcols * y_top + 1;
    // Find a simple path through the different surface corners.
    const IntegralPixel_T tmp1 = (*ptr_integral);
    ptr_integral += dx;
    const IntegralPixel_T tmp2 = (*ptr_integral);
    ptr_integral += integralcols;
    const IntegralPixel_T tmp3 = (*ptr_integral);
    ptr_integral++;
    const IntegralPixel_T tmp4 = (*ptr_integral);
    ptr_integral += dy * integralcols;
    const IntegralPixel_T tmp5 = (*ptr_integral);
    ptr_integral--;
    const IntegralPixel_T tmp6 = (*ptr_integral);
    ptr_integral += integralcols;
    const IntegralPixel_T tmp7 = (*ptr_integral);
    ptr_integral -= dx;
    const IntegralPixel_T tmp8 = (*ptr_integral);
    ptr_integral -= integralcols;
    const IntegralPixel_T tmp9 = (*ptr_integral);
    ptr_integral--;
    const IntegralPixel_T tmp10 = (*ptr_integral);
    ptr_integral -= dy * integralcols;
    const IntegralPixel_T tmp11 = (*ptr_integral);
    ptr_integral++;
    const IntegralPixel_T tmp12 = (*ptr_integral);

    // Assign the weighted surface integrals:
    const IntegralPixel_T upper = (tmp3 - tmp2 + tmp1 - tmp12) * r_y_1_i;
    const IntegralPixel_T middle = (tmp6 - tmp3 + tmp12 - tmp9) * scaling;
    const IntegralPixel_T left = (tmp9 - tmp12 + tmp11 - tmp10) * r_x_1_i;
    const IntegralPixel_T right = (tmp5 - tmp4 + tmp3 - tmp6) * r_x1_i;
    const IntegralPixel_T bottom = (tmp7 - tmp6 + tmp9 - tmp8) * r_y1_i;

    return IntegralPixel_T(
        (ret_val + upper + middle + left + right + bottom) / scaling2);
  }

  // Now the calculation:
  ImgPixel_T* ptr = reinterpret_cast<ImgPixel_T*>(image.data) + x_left
      + imagecols * y_top;
  // First row:
  ret_val = A * IntegralPixel_T(*ptr);
  ptr++;
  const ImgPixel_T* end1 = ptr + dx;
  for (; ptr < end1; ptr++) {
    ret_val += r_y_1_i * IntegralPixel_T(*ptr);
  }
  ret_val += B * IntegralPixel_T(*ptr);
  // Middle ones:
  ptr += imagecols - dx - 1;
  const ImgPixel_T* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1) {
    ret_val += r_x_1_i * IntegralPixel_T(*ptr);
    ptr++;
    const ImgPixel_T* end2 = ptr + dx;
    for (; ptr < end2; ptr++) {
      ret_val += IntegralPixel_T(*ptr) * scaling;
    }
    ret_val += r_x1_i * IntegralPixel_T(*ptr);
  }
  // Last row:
  ret_val += D * IntegralPixel_T(*ptr);
  ptr++;
  const ImgPixel_T* end3 = ptr + dx;
  for (; ptr < end3; ptr++) {
    ret_val += r_y1_i * IntegralPixel_T(*ptr);
  }
  ret_val += C * IntegralPixel_T(*ptr);

  return IntegralPixel_T((ret_val) / scaling2);
}

bool RoiPredicate(const float minX, const float minY, const float maxX,
                  const float maxY, const agast::KeyPoint& keyPt) {
  return (agast::KeyPointX(keyPt) < minX) || (agast::KeyPointX(keyPt) >= maxX)
      || (agast::KeyPointY(keyPt) < minY) || (agast::KeyPointY(keyPt) >= maxY);
}

void BriskDescriptorExtractor::setDescriptorBits(int keypoint_idx,
                                                 const int* values,
                                                 agast::Mat* descriptors) const {
  CHECK_NOTNULL(descriptors);
  unsigned char* ptr = descriptors->data + strings_ * keypoint_idx;

  // Now iterate through all the pairings.
  //brisk::timing::DebugTimer timer_assemble_bits(
      //"1.3 Brisk Extraction: assemble bits (per keypoint)");
  brisk::UINT32_ALIAS* ptr2 = reinterpret_cast<brisk::UINT32_ALIAS*>(ptr);
  const brisk::BriskShortPair* max = shortPairs_ + noShortPairs_;
  int shifter = 0;
  for (brisk::BriskShortPair* iter = shortPairs_; iter < max; ++iter) {
    int t1 = *(values + iter->i);
    int t2 = *(values + iter->j);
    if (t1 > t2) {
      *ptr2 |= ((1) << shifter);
    }  // Else already initialized with zero.
    // Take care of the iterators:
    ++shifter;
    if (shifter == 32) {
      shifter = 0;
      ++ptr2;
    }
  }
  //timer_assemble_bits.Stop();
}

void BriskDescriptorExtractor::setDescriptorBits(
    int keypoint_idx,
    const int* values,
    std::vector<std::bitset<kDescriptorLength> >* descriptors) const {
  CHECK_NOTNULL(descriptors);
  std::bitset<kDescriptorLength>& descriptor = descriptors->at(keypoint_idx);

  // Now iterate through all the pairings.
  // brisk::timing::DebugTimer timer_assemble_bits(
      //"1.3 Brisk Extraction: assemble bits (per keypoint)");
  const brisk::BriskShortPair* max = shortPairs_ + noShortPairs_;
  int shifter = 0;
  for (brisk::BriskShortPair* iter = shortPairs_; iter < max; ++iter) {
    int t1 = *(values + iter->i);
    int t2 = *(values + iter->j);
    if (t1 > t2) {
      descriptor.set(shifter, true);
    }  // Else already initialized with zero.
    ++shifter;
  }
  //timer_assemble_bits.Stop();
}

void BriskDescriptorExtractor::computeImpl(
    const agast::Mat& image, std::vector<agast::KeyPoint>& keypoints,
    std::vector<std::bitset<kDescriptorLength> >& descriptors) const {
  doDescriptorComputation(image, keypoints, descriptors);
}

void BriskDescriptorExtractor::computeImpl(const agast::Mat& image,
                                           std::vector<agast::KeyPoint>& keypoints,
                                           agast::Mat& descriptors) const {
  doDescriptorComputation(image, keypoints, descriptors);
}

void BriskDescriptorExtractor::AllocateDescriptors(size_t count,
                                                   agast::Mat& descriptors) const {
  descriptors = agast::Mat::zeros(count, strings_, CV_8UC1);
}

void BriskDescriptorExtractor::AllocateDescriptors(
    size_t count,
    std::vector<std::bitset<kDescriptorLength> >& descriptors) const {
  descriptors.resize(count);
}

template<typename DESCRIPTOR_CONTAINER>
void BriskDescriptorExtractor::doDescriptorComputation(
    const agast::Mat& image,
    std::vector<agast::KeyPoint>& keypoints,
    DESCRIPTOR_CONTAINER& descriptors) const {
  // Remove keypoints very close to the border.
    size_t ksize = keypoints.size();
    std::vector<int> kscales;  // Remember the scale per keypoint.
    kscales.resize(ksize);
    static const float log2 = 0.693147180559945;
    static const float lb_scalerange = log(scalerange_) / (log2);

    std::vector<agast::KeyPoint> valid_kp;
    std::vector<int> valid_scales;
    valid_kp.reserve(keypoints.size());
    valid_scales.reserve(keypoints.size());

    static const float basicSize06 = basicSize_ * 0.6;
    unsigned int basicscale = 0;
    if (!scaleInvariance)
      basicscale = std::max(
          static_cast<int>(scales_ / lb_scalerange
              * (log(1.45 * basicSize_ / (basicSize06)) / log2) + 0.5),
          0);
    for (size_t k = 0; k < ksize; k++) {
      unsigned int scale;
      if (scaleInvariance) {
        scale = std::max(
            static_cast<int>(scales_ / lb_scalerange
                * (log(agast::KeyPointSize(keypoints[k]) / (basicSize06)) / log2) + 0.5),
            0);
        // Saturate.
        if (scale >= scales_)
          scale = scales_ - 1;
        kscales[k] = scale;
      } else {
        scale = basicscale;
        kscales[k] = scale;
      }
      const int border = sizeList_[scale];
      const int border_x = image.cols - border;
      const int border_y = image.rows - border;
      if (!RoiPredicate(border, border, border_x, border_y, keypoints[k])) {
        valid_kp.push_back(keypoints[k]);
        valid_scales.push_back(kscales[k]);
      }
    }

    keypoints.swap(valid_kp);
    kscales.swap(valid_scales);
    ksize = keypoints.size();

    AllocateDescriptors(keypoints.size(), descriptors);

    // First, calculate the integral image over the whole image:
    // current integral image.

    //brisk::timing::DebugTimer timer_integral_image(
        //"1.0 Brisk Extraction: integral computation");
    cv::Mat _integral;  // The integral image.
    cv::Mat imageScaled;
    if (image.type() == CV_16UC1) {
      IntegralImage16(imageScaled, &_integral);
    } else if (image.type() == CV_8UC1) {
      IntegralImage8(image, &_integral);
    } else {
      throw std::runtime_error("Unsupported image format. Must be CV_16UC1 or CV_8UC1.");
    }
    //timer_integral_image.Stop();

    int* _values = new int[points_];  // For temporary use.

    // Now do the extraction for all keypoints:
    for (size_t k = 0; k < ksize; ++k) {
      int theta;
      agast::KeyPoint& kp = keypoints[k];
      const int& scale = kscales[k];
      int* pvalues = _values;
      const float& x = agast::KeyPointX(kp);
      const float& y = agast::KeyPointY(kp);
      if (agast::KeyPointAngle(kp) == -1) {
        if (!rotationInvariance) {
          // Don't compute the gradient direction, just assign a rotation of 0°.
          theta = 0;
        } else {
          // Get the gray values in the unrotated pattern.
          //brisk::timing::DebugTimer timer_rotation_determination_sample_points(
              //"1.1.1 Brisk Extraction: rotation determination: sample points "
              //"(per keypoint)");
          if (image.type() == CV_8UC1) {
            for (unsigned int i = 0; i < points_; i++) {
              *(pvalues++) = SmoothedIntensity<unsigned char, int>(image, _integral, x, y,
                                                           scale, 0, i);
            }
          } else {
            for (unsigned int i = 0; i < points_; i++) {
              *(pvalues++) = static_cast<int>(65536.0
                  * SmoothedIntensity<float, float>(imageScaled, _integral, x, y,
                                                    scale, 0, i));
            }
          }
          //timer_rotation_determination_sample_points.Stop();
          int direction0 = 0;
          int direction1 = 0;
          // Now iterate through the long pairings.
          //brisk::timing::DebugTimer timer_rotation_determination_gradient(
              //"1.1.2 Brisk Extraction: rotation determination: calculate "
              //"gradient (per keypoint)");
          const brisk::BriskLongPair* max = longPairs_ + noLongPairs_;
          for (brisk::BriskLongPair* iter = longPairs_; iter < max; ++iter) {
            int t1 = *(_values + iter->i);
            int t2 = *(_values + iter->j);
            const int delta_t = (t1 - t2);
            // Update the direction:
            const int tmp0 = delta_t * (iter->weighted_dx) / 1024;
            const int tmp1 = delta_t * (iter->weighted_dy) / 1024;
            direction0 += tmp0;
            direction1 += tmp1;
          }
          //timer_rotation_determination_gradient.Stop();
          kp.angle = atan2(static_cast<float>(direction1),
                           static_cast<float>(direction0)) / M_PI * 180.0;
          theta = static_cast<int>((n_rot_ * agast::KeyPointAngle(kp)) /
                                   (360.0) + 0.5);
          if (theta < 0)
            theta += n_rot_;
          if (theta >= static_cast<int>(n_rot_))
            theta -= n_rot_;
        }
      } else {
        // Figure out the direction:
        if (!rotationInvariance) {
          theta = 0;
        } else {
          theta = static_cast<int>(n_rot_ * (agast::KeyPointAngle(kp) /
              (360.0)) + 0.5);
          if (theta < 0)
            theta += n_rot_;
          if (theta >= static_cast<int>(n_rot_))
            theta -= n_rot_;
        }
      }

      // Now also extract the stuff for the actual direction:
      // Let us compute the smoothed values.
      pvalues = _values;
      // Get the gray values in the rotated pattern.
      //brisk::timing::DebugTimer timer_sample_points(
          //"1.2 Brisk Extraction: sample points (per keypoint)");
      if (image.type() == CV_8UC1) {
        for (unsigned int i = 0; i < points_; i++) {
          *(pvalues++) = SmoothedIntensity<unsigned char, int>(image, _integral, x, y,
                                                       scale, theta, i);
        }
      } else {
        for (unsigned int i = 0; i < points_; i++) {
          *(pvalues++) = static_cast<int>(65536.0
              * SmoothedIntensity<float, float>(imageScaled, _integral, x, y,
                                                scale, theta, i));
        }
      }
      //timer_sample_points.Stop();

      setDescriptorBits(k, _values, &descriptors);
    }
    delete[] _values;
}

int BriskDescriptorExtractor::descriptorSize() const {
  return strings_;
}

int BriskDescriptorExtractor::descriptorType() const {
  return CV_8U;
}

BriskDescriptorExtractor::~BriskDescriptorExtractor() {
  delete[] patternPoints_;
  delete[] shortPairs_;
  delete[] longPairs_;
  delete[] scaleList_;
  delete[] sizeList_;
}
}  // namespace brisk
