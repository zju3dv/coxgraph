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

#ifndef AGAST_WRAP_OPENCV_H_
#define AGAST_WRAP_OPENCV_H_

#define HAVE_OPENCV 1

#if HAVE_OPENCV
#include <agast/glog.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#else
#include <fstream>  // NOLINT
#include <memory>
#include <glog/logging.h>
#include <NCV/NCVLib/feature_detection_and_matching/scale_invariant_feature_detection.h>  // NOLINT
#endif

#if HAVE_OPENCV
namespace agast {
using cv::KeyPoint;
using cv::Point2f;
using cv::Point_;
using cv::imread;
using cv::Mat;

inline float& KeyPointX(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.pt.x;
}
inline const float& KeyPointX(const agast::KeyPoint& keypoint) {
  return keypoint.pt.x;
}
inline float& KeyPointY(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.pt.y;
}
inline const float& KeyPointY(const agast::KeyPoint& keypoint) {
  return keypoint.pt.y;
}
inline float& KeyPointAngle(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.angle;
}
inline const float& KeyPointAngle(const agast::KeyPoint& keypoint) {
  return keypoint.angle;
}
inline float& KeyPointSize(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.size;
}
inline const float& KeyPointSize(const agast::KeyPoint& keypoint) {
  return keypoint.size;
}
inline float& KeyPointResponse(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.response;
}
inline const float& KeyPointResponse(const agast::KeyPoint& keypoint) {
  return keypoint.response;
}
inline int& KeyPointOctave(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.octave;
}
inline const int& KeyPointOctave(const agast::KeyPoint& keypoint) {
  return keypoint.octave;
}
}  // namespace agast
#else

// A minimal opencv subpart with std::shared allocation.
namespace agast {
#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32FC1 CV_MAKETYPE(CV_32F,1)

#define CHECK_VALID_TYPES(type) \
    CHECK(type == CV_8UC1 || type == CV_8UC1 \
          || type == CV_16UC1 || type == CV_16SC1 \
          || type == CV_32SC1 || type == CV_32FC1) << \
     "Only 8/16 bit grayscale is supported.";

struct Mat {
  struct MStep {
    friend struct Mat;
    MStep();
    MStep(size_t s);
    const size_t& operator[](int i) const;
    size_t& operator[](int i);
    operator size_t() const;
    MStep& operator =(size_t s);
   private:
    size_t buf[2];
   protected:
    MStep& operator =(const MStep&);
  };

  struct ImageContainer {
    explicit ImageContainer(unsigned int size) {
      data.reset(new unsigned char[size]);
    }
    ImageContainer(const ImageContainer&) = delete;
    ImageContainer& operator=(const ImageContainer&) const = delete;
    std::unique_ptr<unsigned char[]> data;
  };
  std::shared_ptr<ImageContainer> img;
  int rows;
  int cols;
  int type_;
  MStep step;
  unsigned char* data;
  Mat() : rows(0),
          cols(0),
          type_(CV_8UC1),
          data(nullptr) {
    step[0] = 0;
    step[1] = 0;
  }
  // Wrap user allocated data. Do not take ownership.
  Mat(int _rows, int _cols, int _type, unsigned char* _data) :
    rows(_rows), cols(_cols), type_(_type), data(_data) {
    unsigned int bytedepth = ComputeByteDepth(type_);
    step.buf[0] = cols * bytedepth;
    step.buf[1] = 0;
    // The member img stays uninitialized.
  }
  Mat& operator=(const Mat& other) {
    rows = other.rows;
    cols = other.cols;
    type_ = other.type_;
    step[0] = other.step[0];
    step[1] = other.step[1];
    img = other.img;
    data = other.data;
    return *this;
  }
  Mat(int _rows, int _cols, int _type) {
    create(_rows, _cols, _type);
  }
  void release() {
    if (img) {
      img->data.reset();
    }
    data = nullptr;
  }
  unsigned int elemSize() const {
    return ComputeByteDepth(type_);
  }
  bool isContinuous() const {
    // This implementation does not support submatrices like opencv.
    return true;
  }
  agast::Mat clone() const {
    agast::Mat tmp(rows, cols, type_);
    if (data) {
      unsigned int bytedepth = ComputeByteDepth(type_);
      unsigned int final_size = rows * cols * bytedepth;
      CHECK(tmp.img);
      CHECK(data);
      memcpy(tmp.img->data.get(), data, final_size);
      tmp.data = tmp.img->data.get();
    }
    return tmp;
  }

  void create(int _rows, int _cols, int _type) {
    rows = _rows;
    cols = _cols;
    type_ = _type;
    unsigned int bytedepth = ComputeByteDepth(type_);
    step.buf[0] = cols * bytedepth;
    step.buf[1] = 0;
    CHECK_VALID_TYPES(type_);
    unsigned int final_size = rows * cols * bytedepth;
    img.reset(new ImageContainer(final_size));
    data = img->data.get();
  }

  static agast::Mat zeros(int rows, int cols, int type) {
    agast::Mat tmp(rows, cols, type);
    tmp.setTo<unsigned char>(0);
    return tmp;
  }
  template<typename TYPE>
  void setTo(int value) {
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        at<TYPE>(row, col) = static_cast<TYPE>(value);
      }
    }
  }
  int& type() {
    return type_;
  }
  const int& type() const {
    return type_;
  }
  bool empty() const {
    return !static_cast<bool>(img);
  }
  template<typename _Tp> _Tp& at(int i0=0);
  template<typename _Tp> const _Tp& at(int i0=0) const;

  template<typename _Tp> _Tp& at(int i0, int i1);
  template<typename _Tp> const _Tp& at(int i0, int i1) const;

 private:
  unsigned int ComputeByteDepth(int type) const {
    unsigned int bytedepth = 0;
    if (type == CV_8UC1) {
      bytedepth = 1;
    } else if (type == CV_8SC1) {
      bytedepth = 1;
    } else if (type == CV_16UC1) {
      bytedepth = 2;
    } else if (type == CV_16SC1) {
      bytedepth = 2;
    } else if (type == CV_32SC1) {
      bytedepth = 4;
    } else if (type == CV_32FC1) {
      bytedepth = 4;
    }
    CHECK_NE(bytedepth, 0u) << "Unknown type to compute bytedepth from: "
                            << type;
    return bytedepth;
  }
};

template<typename TYPE>
struct Point_{
  TYPE x;
  TYPE y;
};

typedef Point_<float> Point2f;
typedef Point_<int> Point2i;
// Reads a pgm image from file.
agast::Mat imread(const std::string& filename);
ssize_t imwrite(const agast::Mat& image, const std::string& filename);
typedef NestorCV::SinglePrecisionDoGScaleInvariantDetector::keypoint KeyPoint;

inline float& KeyPointX(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.xs;
}
inline const float& KeyPointX(const agast::KeyPoint& keypoint) {
  return keypoint.xs;
}
inline float& KeyPointY(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.ys;
}
inline const float& KeyPointY(const agast::KeyPoint& keypoint) {
  return keypoint.ys;
}
inline float& KeyPointAngle(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.angle;
}
inline const float& KeyPointAngle(const agast::KeyPoint& keypoint) {
  return keypoint.angle;
}
inline float& KeyPointSize(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.sigma;
}
inline const float& KeyPointSize(const agast::KeyPoint& keypoint) {
  return keypoint.sigma;
}
inline float& KeyPointResponse(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.edge_score;
}
inline const float& KeyPointResponse(const agast::KeyPoint& keypoint) {
  return keypoint.edge_score;
}
inline int& KeyPointOctave(agast::KeyPoint& keypoint) {  // NOLINT
  return keypoint.octave;
}
inline const int& KeyPointOctave(const agast::KeyPoint& keypoint) {
  return keypoint.octave;
}
}  // namespace agast

#endif  // HAVE_OPENCV

#include "./internal/wrap-opencv-inl.h"
#endif  // AGAST_WRAP_OPENCV_H_
