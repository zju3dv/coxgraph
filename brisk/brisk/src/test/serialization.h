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
#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include <agast/wrap-opencv.h>
#include <agast/glog.h>

namespace serialization {

template<class TYPE>
void Serialize(
    const TYPE& value, std::ofstream* out,
    typename std::enable_if<std::is_integral<TYPE>::value>::type* = 0) {
  CHECK_NOTNULL(out);
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template<class TYPE>
void DeSerialize(TYPE* value, std::ifstream* in,
                 typename std::enable_if<std::is_integral<TYPE>::value>::type* =
                     0) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  in->read(reinterpret_cast<char*>(value), sizeof(*value));
}

template<class TYPE>
void Serialize(
    const TYPE& value, std::ofstream* out,
    typename std::enable_if<std::is_floating_point<TYPE>::value>::type* = 0) {
  CHECK_NOTNULL(out);
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template<class TYPE>
void DeSerialize(
    TYPE* value, std::ifstream* in,
    typename std::enable_if<std::is_floating_point<TYPE>::value>::type* = 0) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  in->read(reinterpret_cast<char*>(value), sizeof(*value));
}

void Serialize(const uint32_t& value, std::ofstream* out);

void DeSerialize(uint32_t* value, std::ifstream* in);

void Serialize(const agast::Mat& mat, std::ofstream* out);

void DeSerialize(agast::Mat* mat, std::ifstream* in);

void Serialize(const agast::Point2f& pt, std::ofstream* out);

void Serialize(const agast::KeyPoint& pt, std::ofstream* out);

void DeSerialize(agast::KeyPoint* pt, std::ifstream* in);

void Serialize(const std::string& value, std::ofstream* out);

void DeSerialize(std::string* value, std::ifstream* in);

template<typename TYPE>
void DeSerialize(agast::Point_<TYPE>* pt, std::ifstream* in) {
  CHECK_NOTNULL(pt);
  CHECK_NOTNULL(in);
  DeSerialize(&pt->x, in);
  DeSerialize(&pt->y, in);
}

template<typename TYPEA, typename TYPEB>
void Serialize(const std::pair<TYPEA, TYPEB>& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  Serialize(value.first, out);
  Serialize(value.second, out);
}

template<typename TYPEA, typename TYPEB>
void DeSerialize(std::pair<TYPEA, TYPEB>* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  DeSerialize(&value->first, in);
  DeSerialize(&value->second, in);
}

template<typename TYPEA, typename TYPEB>
void Serialize(const std::map<TYPEA, TYPEB>& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  uint32_t length = value.size();
  Serialize(length, out);
  for (const std::pair<TYPEA, TYPEB>& entry : value) {
    Serialize(entry, out);
  }
}

template<typename TYPEA, typename TYPEB>
void DeSerialize(std::map<TYPEA, TYPEB>* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  value->clear();
  uint32_t length;
  DeSerialize(&length, in);
  for (uint32_t i = 0; i < length; ++i) {
    std::pair<TYPEA, TYPEB> entry;
    DeSerialize(&entry, in);
    value->insert(entry);
  }
}

// TODO(slynen): Merge the templates for vector and list using template template
// parameters.
template<typename T>
void Serialize(const std::vector<T>& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  uint32_t length = value.size();
  Serialize(length, out);
  for (const T& entry : value) {
    Serialize(entry, out);
  }
}

template<typename T>
void DeSerialize(std::vector<T>* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  value->clear();
  uint32_t length;
  DeSerialize(&length, in);
  value->resize(length);
  for (T& entry : *value) {
    DeSerialize(&entry, in);
  }
}

template<typename T>
void Serialize(const std::list<T>& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  uint32_t length = value.size();
  Serialize(length, out);
  for (const T& entry : value) {
    Serialize(entry, out);
  }
}

template<typename T>
void DeSerialize(std::list<T>* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  value->clear();
  uint32_t length;
  DeSerialize(&length, in);
  value->resize(length);
  for (T& entry : *value) {
    DeSerialize(&entry, in);
  }
}

}  // namespace serialization

#endif  // SERIALIZATION_H_
