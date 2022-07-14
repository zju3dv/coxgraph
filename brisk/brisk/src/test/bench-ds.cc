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

#include "./bench-ds.h"

namespace brisk {

#ifdef __ARM_NEON
std::string DescriptorToString(const uint8x16_t* d, int num128Words) {
#else
  std::string DescriptorToString(const __m128i * d, int num128Words) {
#endif
  std::stringstream ss;
  ss << "[";
  const unsigned int __attribute__ ((__may_alias__))* data =
      reinterpret_cast<const unsigned int __attribute__ ((__may_alias__))*>(d);
  for (int bit = 0; bit < num128Words * 128; ++bit) {
    ss << (data[bit >> 5] & (1 << (bit & 31)) ? "1 " : "0 ");
  };
  return ss.str();
}

void Serialize(const Blob& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  serialization::Serialize(value.size_, out);
  out->write(reinterpret_cast<const char*>(value.verification_data_.get()),
             value.size_);
}

void DeSerialize(Blob* value, std::ifstream* in) {
  CHECK_NOTNULL(in);
  CHECK_NOTNULL(value);
  serialization::DeSerialize(&value->size_, in);
  value->verification_data_.reset(new unsigned char[value->size_]);
  in->read(reinterpret_cast<char*>(value->verification_data_.get()),
           value->size_);
}

void Serialize(const DatasetEntry& value, std::ofstream* out) {
  CHECK_NOTNULL(out);
  serialization::Serialize(value.path_, out);
  serialization::Serialize(value.imgGray_, out);
  serialization::Serialize(value.keypoints_, out);
  serialization::Serialize(value.descriptors_, out);
  serialization::Serialize(value.userdata_, out);
}

void DeSerialize(DatasetEntry* value, std::ifstream* in) {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(in);
  try {
    serialization::DeSerialize(&value->path_, in);
    serialization::DeSerialize(&value->imgGray_, in);
    serialization::DeSerialize(&value->keypoints_, in);
    serialization::DeSerialize(&value->descriptors_, in);
    serialization::DeSerialize(&value->userdata_, in);
  } catch (const std::ifstream::failure& e) {
    CHECK(false) << "Failed to load DatasetEntry " + std::string(e.what());
  }
}

}  // namespace brisk
