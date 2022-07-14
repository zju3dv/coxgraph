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

#include <bitset>

#include <brisk/internal/hamming.h>
#include <agast/glog.h>
#include <gtest/gtest.h>

#ifndef TEST
#define TEST(a, b) int Test_##a##_##b()
#endif
  enum {
    num_128_words = 1,
    array_length = 16 * num_128_words
  };

unsigned int PopCountOfXor(const unsigned char* data1,
                           const unsigned char* data2) {
  unsigned int count = 0;
  for (int i = 0; i < array_length; ++i) {
    unsigned char xor_result = data1[i]^data2[i];
    for (int bit = 0; bit < 8; ++bit) {
      count += static_cast<bool>(xor_result & 1 << bit);
    }
  }
  return count;
}


TEST(Brisk, PopCount) {
  unsigned char data1[array_length];
  unsigned char data2[array_length];

  memset(data1, 0, array_length);
  memset(data2, 0, array_length);

  data1[0] = 0x5;
  data1[3] = 0x2;
  data1[6] = 0x34;
  data1[8] = 0x7;
  data1[10] = 0x23;
  data1[13] = 0x45;
  data1[15] = 0x78;

  data2[0] = 0x22;
  data2[3] = 0x78;
  data2[6] = 0x12;
  data2[8] = 0x32;
  data2[10] = 0x1;
  data2[13] = 0x23;
  data2[15] = 0x75;

  brisk::Hamming popcnt;
#if __ARM_NEON
  const uint8x16_t* signature1 = reinterpret_cast<const uint8x16_t*>(data1);
  const uint8x16_t* signature2 = reinterpret_cast<const uint8x16_t*>(data2);
  unsigned int cnt = popcnt.NEONPopcntofXORed(signature1, signature2,
                                              num_128_words);
#else
  const __m128i* signature1 = reinterpret_cast<const __m128i*>(data1);
  const __m128i* signature2 = reinterpret_cast<const __m128i*>(data2);
  unsigned int cnt = popcnt.SSSE3PopcntofXORed(signature1, signature2,
                                               num_128_words);
#endif  // __ARM_NEON
  unsigned int verification_result = PopCountOfXor(data1, data2);
  ASSERT_EQ(cnt, verification_result);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
