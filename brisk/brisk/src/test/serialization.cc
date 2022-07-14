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

#include "./serialization.h"

namespace serialization {

void Serialize(const agast::Mat& mat, std::ofstream* out) {
  CHECK_NOTNULL(out);
  agast::Mat mat_cont;
  if (!mat.isContinuous()) {
    mat_cont = mat.clone();
  } else {
    mat_cont = mat;
  }
  int type = mat_cont.type();
  int element_size = mat_cont.elemSize();

  Serialize(mat_cont.rows, out);
  Serialize(mat_cont.cols, out);
  Serialize(type, out);
  Serialize(element_size, out);

  out->write(reinterpret_cast<char*>(mat_cont.data),
             element_size * mat_cont.rows * mat_cont.cols);
}

void DeSerialize(agast::Mat* mat, std::ifstream* in) {
  CHECK_NOTNULL(mat);
  CHECK_NOTNULL(in);
  int rows;
  int cols;
  int type;
  int element_size;
  DeSerialize(&rows, in);
  DeSerialize(&cols, in);
  DeSerialize(&type, in);
  DeSerialize(&element_size, in);
  mat->create(rows, cols, type);
  in->read(reinterpret_cast<char*>(mat->data), element_size * rows * cols);
}

void Serialize(const agast::Point2f& pt, std::ofstream* out) {
  CHECK_NOTNULL(out);
  Serialize(pt.x, out);
  Serialize(pt.y, out);
}

void Serialize(const agast::KeyPoint& pt, std::ofstream* out) {
  CHECK_NOTNULL(out);
  Serialize(agast::KeyPointAngle(pt), out);
#ifdef HAVE_OPENCV
  Serialize(pt.class_id, out);
#else
  int class_id = 0;  // Discarded.
  Serialize(class_id, out);
#endif  // HAVE_OPENCV
  Serialize(pt.octave, out);
  Serialize(agast::KeyPointX(pt), out);
  Serialize(agast::KeyPointY(pt), out);
  Serialize(agast::KeyPointResponse(pt), out);
  Serialize(agast::KeyPointSize(pt), out);
}

void DeSerialize(agast::KeyPoint* pt, std::ifstream* in) {
  CHECK_NOTNULL(pt);
  CHECK_NOTNULL(in);
  DeSerialize(&agast::KeyPointAngle(*pt), in);
#ifdef HAVE_OPENCV
  DeSerialize(&pt->class_id, in);
#else
  int class_id = 0;  // Discarded.
  DeSerialize(&class_id, in);
#endif
  DeSerialize(&agast::KeyPointOctave(*pt), in);
  DeSerialize(&agast::KeyPointX(*pt), in);
  DeSerialize(&agast::KeyPointY(*pt), in);
  DeSerialize(&agast::KeyPointResponse(*pt), in);
  DeSerialize(&agast::KeyPointSize(*pt), in);
}

void Serialize(const std::string& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  uint32_t length = value.size();
  Serialize(length, out);
  out->write(reinterpret_cast<const char*>(value.data()),
             length * sizeof(value[0]));
}

void DeSerialize(std::string* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  uint32_t length;
  DeSerialize(&length, in);
  value->resize(length);
  std::unique_ptr<char[]> mem(new char[length + 1]);
  in->read(mem.get(), length * sizeof(mem.get()[0]));
  mem[length] = '\0';
  *value = std::string(mem.get());
}

void Serialize(const uint32_t& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void DeSerialize(uint32_t* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  in->read(reinterpret_cast<char*>(value), sizeof(*value));
}
}  // namespace serialization
