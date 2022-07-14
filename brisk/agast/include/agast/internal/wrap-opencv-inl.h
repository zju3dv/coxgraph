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

#ifndef AGAST_WRAP_OPENCV_INL_H_
#define AGAST_WRAP_OPENCV_INL_H_

#include <agast/wrap-opencv.h>

#if !HAVE_OPENCV
namespace agast {
inline Mat::MStep::MStep() {
  buf[0] = buf[1] = 0;
}
inline Mat::MStep::MStep(size_t s) {
  buf[0] = s;
  buf[1] = 0;
}
inline const size_t& Mat::MStep::operator[](int i) const {
  return buf[i];
}
inline size_t& Mat::MStep::operator[](int i) {
  return buf[i];
}
inline Mat::MStep::operator size_t() const {
  return buf[0];
}
inline Mat::MStep& Mat::MStep::operator =(size_t s) {
  buf[0] = s;
  return *this;
}

template<typename _Tp> inline _Tp& Mat::at(int i0, int i1) {
  return ((_Tp*) (data + step.buf[0] * i0))[i1];
}

template<typename _Tp> inline const _Tp& Mat::at(int i0, int i1) const {
  return ((const _Tp*) (data + step.buf[0] * i0))[i1];
}

template<typename _Tp> inline _Tp& Mat::at(int i0) {
  if (isContinuous() || rows == 1)
    return ((_Tp*) data)[i0];
  if (cols == 1)
    return *(_Tp*) (data + step.buf[0] * i0);
  int i = i0 / cols, j = i0 - i * cols;
  return ((_Tp*) (data + step.buf[0] * i))[j];
}

template<typename _Tp> inline const _Tp& Mat::at(int i0) const {
  if (isContinuous() || rows == 1)
    return ((const _Tp*) data)[i0];
  if (cols == 1)
    return *(const _Tp*) (data + step.buf[0] * i0);
  int i = i0 / cols, j = i0 - i * cols;
  return ((const _Tp*) (data + step.buf[0] * i))[j];
}
}  // namespace agast
#endif  // !HAVE_OPENCV
#endif  // AGAST_WRAP_OPENCV_INL_H_
