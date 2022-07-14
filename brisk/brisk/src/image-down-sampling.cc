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

#include <brisk/internal/image-down-sampling.h>
#include <brisk/internal/macros.h>
#include <agast/glog.h>

namespace {
#ifdef __ARM_NEON
inline uint8x16_t shiftrightonebyte(uint8x16_t& data) {
  uint64x2_t newval = vreinterpretq_u64_u8(data);
  uint64x2_t shiftval = vshrq_n_u64(newval, 8);
  uint8x16_t shiftval8 = vreinterpretq_u8_u64(shiftval);
  uint8_t lostbyte = vgetq_lane_u8(data, 9);
  shiftval8 = vsetq_lane_u8(lostbyte, shiftval8, 8);
  return shiftval8;
}
#endif
}

namespace brisk {
void Halfsample16(const agast::Mat& srcimg, agast::Mat& dstimg) {
#ifdef __ARM_NEON
  static_cast<void>(srcimg);
  static_cast<void>(dstimg);
  CHECK(false) << "HalfSample16 not implemented for NEON.";
#else
  // Make sure the destination image is of the right size:
  CHECK_EQ(srcimg.cols / 2, dstimg.cols);
  CHECK_EQ(srcimg.rows / 2, dstimg.rows);
  CHECK_EQ(srcimg.type(), CV_16UC1);

  const __m128i ones = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);

  const int colsMax = (srcimg.cols / 2) * 2 - 16;
  const int rows = (srcimg.rows / 2) * 2 - 1;

  for (int y = 0; y < rows; y += 2) {
    bool end = false;
    int x_store = 0;
    for (int x = 0; x <= colsMax; x += 16) {
      assert(x + 15 < srcimg.cols);
      assert(y + 1 < srcimg.rows);
      // Load block of four.
      __m128i i00 = _mm_set_epi16(srcimg.at<uint16_t>(y, x + 14),
                                  srcimg.at<uint16_t>(y, x + 12),
                                  srcimg.at<uint16_t>(y, x + 10),
                                  srcimg.at<uint16_t>(y, x + 8),
                                  srcimg.at<uint16_t>(y, x + 6),
                                  srcimg.at<uint16_t>(y, x + 4),
                                  srcimg.at<uint16_t>(y, x + 2),
                                  srcimg.at<uint16_t>(y, x));
      __m128i i01 = _mm_set_epi16(srcimg.at<uint16_t>(y, x + 15),
                                  srcimg.at<uint16_t>(y, x + 13),
                                  srcimg.at<uint16_t>(y, x + 11),
                                  srcimg.at<uint16_t>(y, x + 9),
                                  srcimg.at<uint16_t>(y, x + 7),
                                  srcimg.at<uint16_t>(y, x + 5),
                                  srcimg.at<uint16_t>(y, x + 3),
                                  srcimg.at<uint16_t>(y, x + 1));
      const int y1 = y + 1;
      __m128i i10 = _mm_set_epi16(srcimg.at<uint16_t>(y1, x + 14),
                                  srcimg.at<uint16_t>(y1, x + 12),
                                  srcimg.at<uint16_t>(y1, x + 10),
                                  srcimg.at<uint16_t>(y1, x + 8),
                                  srcimg.at<uint16_t>(y1, x + 6),
                                  srcimg.at<uint16_t>(y1, x + 4),
                                  srcimg.at<uint16_t>(y1, x + 2),
                                  srcimg.at<uint16_t>(y1, x));
      __m128i i11 = _mm_set_epi16(srcimg.at<uint16_t>(y1, x + 15),
                                  srcimg.at<uint16_t>(y1, x + 13),
                                  srcimg.at<uint16_t>(y1, x + 11),
                                  srcimg.at<uint16_t>(y1, x + 9),
                                  srcimg.at<uint16_t>(y1, x + 7),
                                  srcimg.at<uint16_t>(y1, x + 5),
                                  srcimg.at<uint16_t>(y1, x + 3),
                                  srcimg.at<uint16_t>(y1, x + 1));

      // Average.
      i10 = _mm_adds_epu16(i10, ones);
      __m128i result1 = _mm_avg_epu16(i00, i01);
      i10 = _mm_adds_epu16(i10, ones);
      __m128i result2 = _mm_avg_epu16(i10, i11);

      __m128i result = _mm_avg_epu16(result1, result2);

      // Store.
      assert(x_store + 7 < dstimg.cols);
      assert(y / 2 < dstimg.rows);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(
          &(dstimg.at<uint16_t>(y / 2, x_store))), result);

      x_store += 8;

      if (end)
        break;
      if (x + 16 >= colsMax) {
        x = colsMax - 16;
        x_store = dstimg.cols - 8;
        end = true;
      }
    }
  }
#endif  // __ARM_NEON
}

// Half sampling.
void Halfsample8(const agast::Mat& srcimg, agast::Mat& dstimg) {
const uint16_t leftoverCols = ((srcimg.cols % 16) / 2);
const bool noleftover = (srcimg.cols % 16) == 0;

// Make sure the destination image is of the right size:
CHECK_EQ(srcimg.cols / 2, dstimg.cols);
CHECK_EQ(srcimg.rows / 2, dstimg.rows);
#ifdef __ARM_NEON
  // Mask needed later:
  uint8_t tmpmask[16] = {0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
    0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00};
  uint8x16_t mask = vld1q_u8(&tmpmask[0]);
  // To be added in order to make successive averaging correct:
  uint8_t tmpones[16] = {0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
      0x1, 0x1, 0x1, 0x1, 0x1};
  uint8x16_t ones = vld1q_u8(&tmpones[0]);

  // Data pointers:
  const uint8x16_t* p1 = reinterpret_cast<const uint8x16_t*>(srcimg.data);
  const uint8x16_t* p2 = reinterpret_cast<const uint8x16_t*>(
      srcimg.data + srcimg.cols);
  uint8x16_t* p_dest = reinterpret_cast<uint8x16_t*>(dstimg.data);

  unsigned char* p_dest_char;

  // Size:
  const unsigned int hsize = srcimg.cols / 16;
  const uint8x16_t* p_end = reinterpret_cast<const uint8x16_t*>
      (srcimg.data + (srcimg.cols * srcimg.rows) - leftoverCols);
  unsigned int row = 0;
  const unsigned int end = hsize / 2;
  bool half_end;
  if (hsize % 2 == 0)
    half_end = false;
  else
    half_end = true;
  while (p2 < p_end) {
    for (unsigned int i = 0; i < end; ++i) {
      // Load the two blocks of memory:
      uint8x16_t upper = vld1q_u8(reinterpret_cast<const uint8_t*>(p1));
      uint8x16_t lower = vld1q_u8(reinterpret_cast<const uint8_t*>(p2));

      uint8x16_t result1 = vqaddq_u8(upper, ones);
      result1 = vrhaddq_u8(upper, lower);  // Average - halving add.

      ++p1;
      ++p2;

      // Load the two blocks of memory:
      upper = vld1q_u8(reinterpret_cast<const uint8_t*>(p1));
      lower = vld1q_u8(reinterpret_cast<const uint8_t*>(p2));
      uint8x16_t result2 = vqaddq_u8(upper, ones);
      result2 = vrhaddq_u8(upper, lower);
      // Calculate the shifted versions:

      uint8x16_t result1_shifted = shiftrightonebyte(result1);

      uint8x16_t result2_shifted = shiftrightonebyte(result2);

      // Pack:
      uint8x16_t result = vcombine_u8(
          // AND and saturate to uint8.
          vqmovn_u16(vreinterpretq_u16_u8(vandq_u8(result1, mask))),
          // Combine.
          vqmovn_u16(vreinterpretq_u16_u8(vandq_u8(result2, mask))));

      uint8x16_t result_shifted = vcombine_u8(
          // AND and saturate to uint8.
          vqmovn_u16(vreinterpretq_u16_u8(vandq_u8(result1_shifted, mask))),
          vqmovn_u16(vreinterpretq_u16_u8(vandq_u8(result2_shifted, mask))));
      // Average for the second time:

      result = vrhaddq_u8(result, result_shifted);

      // Store.
      vst1q_u8(reinterpret_cast<uint8_t*>(p_dest), result);

      ++p1;
      ++p2;
      ++p_dest;
    }
    // If we are not at the end of the row, do the rest:
    if (half_end) {
      // Load the two blocks of memory:
      uint8x16_t upper = vld1q_u8(reinterpret_cast<const uint8_t*>(p1));
      uint8x16_t lower = vld1q_u8(reinterpret_cast<const uint8_t*>(p2));

      uint8x16_t result1 = vqaddq_u8(upper, ones);
      result1 = vrhaddq_u8(upper, lower);

      // Increment the pointers:
      ++p1;
      ++p2;

      // Compute horizontal pairwise average and store:
      p_dest_char = reinterpret_cast<unsigned char*>(p_dest);
      const UCHAR_ALIAS* result = reinterpret_cast<UCHAR_ALIAS*>(&result1);
      for (unsigned int j = 0; j < 8; ++j) {
        *(p_dest_char++) = (*(result + 2 * j) + *(result + 2 * j + 1)) / 2;
      }
    } else {
      p_dest_char = reinterpret_cast<unsigned char*>(p_dest);
    }

    if (noleftover) {
      ++row;
      p_dest = reinterpret_cast<uint8x16_t*>(dstimg.data + row * dstimg.cols);
      p1 = reinterpret_cast<const uint8x16_t*>(srcimg.data + 2 * row * srcimg.cols);
      p2 = p1 + hsize;
    } else {
      const unsigned char* p1_src_char =
          reinterpret_cast<const unsigned char*>(p1);
      const unsigned char* p2_src_char =
          reinterpret_cast<const unsigned char*>(p2);
      for (unsigned int k = 0; k < leftoverCols; ++k) {
        uint16_t tmp = p1_src_char[2*k] + p1_src_char[2*k + 1]
        + p2_src_char[2*k] + p2_src_char[2*k + 1];
        *(p_dest_char++) = static_cast<unsigned char>((tmp+2) / 4);
      }
      // Done with the two rows:
      ++row;
      p_dest = reinterpret_cast<uint8x16_t*>(dstimg.data + row * dstimg.cols);
      p1 = reinterpret_cast<const uint8x16_t*>(srcimg.data + 2 * row *
                                               srcimg.cols);
      p2 = reinterpret_cast<const uint8x16_t*>(srcimg.data + (2 * row + 1) *
          srcimg.cols);
    }
  }
#else
  // Mask needed later:
  __m128i mask = _mm_set_epi32(0x00FF00FF, 0x00FF00FF, 0x00FF00FF,
                                        0x00FF00FF);
  // To be added in order to make successive averaging correct:
  __m128i ones = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1);

  // Data pointers:
  __m128i* p1 = reinterpret_cast<__m128i*>(srcimg.data);
  __m128i* p2 = reinterpret_cast<__m128i*>(srcimg.data+srcimg.cols);
  __m128i* p_dest = reinterpret_cast<__m128i*>(dstimg.data);
  unsigned char* p_dest_char;

  // Size:
  const unsigned int hsize = srcimg.cols / 16;
  const __m128i* p_end = reinterpret_cast<const __m128i*>
      (srcimg.data + (srcimg.cols * srcimg.rows) - leftoverCols);
  unsigned int row = 0;
  const unsigned int end = hsize / 2;
  bool half_end;
  if (hsize % 2 == 0)
    half_end = false;
  else
    half_end = true;
  while (p2 < p_end) {
    for (unsigned int i = 0; i < end; ++i) {
      // Load the two blocks of memory:
      __m128i upper;
      __m128i lower;
      if (noleftover) {
        upper = _mm_load_si128(p1);
        lower = _mm_load_si128(p2);
      } else {
        upper = _mm_loadu_si128(p1);
        lower = _mm_loadu_si128(p2);
      }

      __m128i result1 = _mm_adds_epu8(upper, ones);
      result1 = _mm_avg_epu8(upper, lower);

      // Increment the pointers:
      ++p1;
      ++p2;

      // Load the two blocks of memory:
      upper = _mm_loadu_si128(p1);
      lower = _mm_loadu_si128(p2);
      __m128i result2 = _mm_adds_epu8(upper, ones);
      result2 = _mm_avg_epu8(upper, lower);
      // Calculate the shifted versions:
      __m128i result1_shifted = _mm_srli_si128(result1, 1);
      __m128i result2_shifted = _mm_srli_si128(result2, 1);
      // Pack:
      __m128i result = _mm_packus_epi16(_mm_and_si128(result1, mask),
                                        _mm_and_si128(result2, mask));
      __m128i result_shifted = _mm_packus_epi16(
          _mm_and_si128(result1_shifted, mask),
          _mm_and_si128(result2_shifted, mask));
      // Average for the second time:
      result = _mm_avg_epu8(result, result_shifted);

      // Store to memory
      _mm_storeu_si128(p_dest, result);

      // Increment the pointers:
      ++p1;
      ++p2;
      ++p_dest;
    }
    // If we are not at the end of the row, do the rest:
    if (half_end) {
      // Load the two blocks of memory:
      __m128i upper;
      __m128i lower;
      if (noleftover) {
        upper = _mm_load_si128(p1);
        lower = _mm_load_si128(p2);
      } else {
        upper = _mm_loadu_si128(p1);
        lower = _mm_loadu_si128(p2);
      }

      __m128i result1 = _mm_adds_epu8(upper, ones);
      result1 = _mm_avg_epu8(upper, lower);

      // Increment the pointers:
      ++p1;
      ++p2;

      // Compute horizontal pairwise average and store.
      p_dest_char = reinterpret_cast<unsigned char*>(p_dest);
      const unsigned char* result = reinterpret_cast<unsigned char*>(&result1);
      for (unsigned int j = 0; j < 8; ++j) {
        *(p_dest_char++) = (*(result + 2 * j) + *(result + 2 * j + 1)) / 2;
      }
    } else {
      p_dest_char = reinterpret_cast<unsigned char*>(p_dest);
    }

    if (noleftover) {
      ++row;
      p_dest = reinterpret_cast<__m128i*>(dstimg.data + row * dstimg.cols);
      p1 = reinterpret_cast<__m128i*>(srcimg.data + 2 * row * srcimg.cols);
      p2 = p1 + hsize;
    } else {
      const unsigned char* p1_src_char = reinterpret_cast<unsigned char*>(p1);
      const unsigned char* p2_src_char = reinterpret_cast<unsigned char*>(p2);
      for (unsigned int k = 0; k < leftoverCols; ++k) {
        uint16_t tmp = p1_src_char[2*k] + p1_src_char[2*k + 1]
        + p2_src_char[2*k] + p2_src_char[2*k + 1];
        *(p_dest_char++) = static_cast<unsigned char>((tmp+2) / 4);
      }
      // Done with the two rows:
      ++row;
      p_dest = reinterpret_cast<__m128i*>(dstimg.data + row * dstimg.cols);
      p1 = reinterpret_cast<__m128i*>(srcimg.data + 2 * row * srcimg.cols);
      p2 = reinterpret_cast<__m128i*>(
          srcimg.data + (2 * row + 1) * srcimg.cols);
    }
  }
#endif  // __ARM_NEON
}

void Twothirdsample16(const agast::Mat& srcimg, agast::Mat& dstimg) {
#ifdef __ARM_NEON
  static_cast<void>(srcimg);
  static_cast<void>(dstimg);
  CHECK(false) << "Twothirdsample16 not implemented for NEON";
#else
  assert(srcimg.type() == CV_16UC1);

  // Make sure the destination image is of the right size:
  assert((srcimg.cols / 3) * 2 == dstimg.cols);
  assert((srcimg.rows / 3) * 2 == dstimg.rows);

  const int colsMax = (srcimg.cols / 3) * 3 - 12;
  const int rows = (srcimg.rows / 3) * 3 - 2;

  for (int y = 0; y < rows; y += 3) {
    bool end = false;
    int x_store = 0;
    for (int x = 0; x <= colsMax; x += 12) {
      assert(x + 11 < srcimg.cols);
      assert(y + 2 < srcimg.rows);

      // Use 2 3x3 blocks.
      const int y1 = y + 1;
      const int y2 = y1 + 1;

      // Assemble epi32 registers.
      // Top row.
      __m128i i0_corners = _mm_set_epi32(srcimg.at<uint16_t>(y, x + 5),
                                         srcimg.at<uint16_t>(y, x + 3),
                                         srcimg.at<uint16_t>(y, x + 2),
                                         srcimg.at<uint16_t>(y, x));
      i0_corners = _mm_slli_epi32(i0_corners, 2);  // * 4.
      const uint32_t m01 = srcimg.at<uint16_t>(y, x + 1) << 1;  // * 2.
      const uint32_t m04 = srcimg.at<uint16_t>(y, x + 4) << 1;  // * 2.
      __m128i i0_middle = _mm_set_epi32(m04, m04, m01, m01);

      // Middle row.
      __m128i i1_leftright = _mm_set_epi32(srcimg.at<uint16_t>(y1, x + 5),
                                           srcimg.at<uint16_t>(y1, x + 3),
                                           srcimg.at<uint16_t>(y1, x + 2),
                                           srcimg.at<uint16_t>(y1, x));
      i1_leftright = _mm_slli_epi32(i1_leftright, 1);  // * 2.
      const uint32_t m11 = srcimg.at<uint16_t>(y1, x + 1);
      const uint32_t m14 = srcimg.at<uint16_t>(y1, x + 4);
      __m128i i1_middle = _mm_set_epi32(m14, m14, m11, m11);

      // Bottom row.
      __m128i i2_corners = _mm_set_epi32(srcimg.at<uint16_t>(y2, x + 5),
                                         srcimg.at<uint16_t>(y2, x + 3),
                                         srcimg.at<uint16_t>(y2, x + 2),
                                         srcimg.at<uint16_t>(y2, x));
      i2_corners = _mm_slli_epi32(i2_corners, 2);  // *4
      const uint32_t m21 = srcimg.at<uint16_t>(y2, x + 1) << 1;  // *2
      const uint32_t m24 = srcimg.at<uint16_t>(y2, x + 4) << 1;  // *2
      __m128i i2_middle = _mm_set_epi32(m24, m24, m21, m21);

      // Average.
      __m128i result1 = _mm_add_epi32(i1_middle, i1_leftright);
      // Top output row.
      __m128i result0 = _mm_add_epi32(i0_corners, i0_middle);
      result0 = _mm_add_epi32(result0, result1);

      // Bottom output row.
      __m128i result2 = _mm_add_epi32(i2_corners, i2_middle);
      result2 = _mm_add_epi32(result2, result1);

      // Assemble epi32 registers---right blocks.
      const int xp = x + 6;
      // Top row.
      i0_corners = _mm_set_epi32(srcimg.at<uint16_t>(y, xp + 5),
                                 srcimg.at<uint16_t>(y, xp + 3),
                                 srcimg.at<uint16_t>(y, xp + 2),
                                 srcimg.at<uint16_t>(y, xp));
      i0_corners = _mm_slli_epi32(i0_corners, 2);  // *4
      const uint32_t m01p = srcimg.at<uint16_t>(y, xp + 1) << 1;  // *2
      const uint32_t m04p = srcimg.at<uint16_t>(y, xp + 4) << 1;  // *2
      i0_middle = _mm_set_epi32(m04p, m04p, m01p, m01p);

      // Middle row.
      i1_leftright = _mm_set_epi32(srcimg.at<uint16_t>(y1, xp + 5),
                                   srcimg.at<uint16_t>(y1, xp + 3),
                                   srcimg.at<uint16_t>(y1, xp + 2),
                                   srcimg.at<uint16_t>(y1, xp));
      i1_leftright = _mm_slli_epi32(i1_leftright, 1);  // *2
      const uint32_t m11p = srcimg.at<uint16_t>(y1, xp + 1);
      const uint32_t m14p = srcimg.at<uint16_t>(y1, xp + 4);
      i1_middle = _mm_set_epi32(m14p, m14p, m11p, m11p);

      // Bottom row.
      i2_corners = _mm_set_epi32(srcimg.at<uint16_t>(y2, xp + 5),
                                 srcimg.at<uint16_t>(y2, xp + 3),
                                 srcimg.at<uint16_t>(y2, xp + 2),
                                 srcimg.at<uint16_t>(y2, xp));
      i2_corners = _mm_slli_epi32(i2_corners, 2);  // *4
      const uint32_t m21p = srcimg.at<uint16_t>(y2, xp + 1) << 1;  // *2
      const uint32_t m24p = srcimg.at<uint16_t>(y2, xp + 4) << 1;  // *2
      i2_middle = _mm_set_epi32(m24p, m24p, m21p, m21p);

      // Average.
      result1 = _mm_add_epi32(i1_middle, i1_leftright);
      // Top output row.
      __m128i result0p = _mm_add_epi32(i0_corners, i0_middle);
      result0p = _mm_add_epi32(result0p, result1);

      // Bottom output row.
      __m128i result2p = _mm_add_epi32(i2_corners, i2_middle);
      result2p = _mm_add_epi32(result2p, result1);

      // Divide by 9 - not sure if this is very safe...
      (reinterpret_cast<INT32_ALIAS*>(&result0p))[0] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0p))[1] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0p))[2] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0p))[3] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2p))[0] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2p))[1] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2p))[2] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2p))[3] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0))[0] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0))[1] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0))[2] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result0))[3] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2))[0] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2))[1] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2))[2] /= 9;
      (reinterpret_cast<INT32_ALIAS*>(&result2))[3] /= 9;

      // Pack.
      __m128i store0 = _mm_packs_epi32(result0, result0p);
      __m128i store2 = _mm_packs_epi32(result2, result2p);

      // Store.
      assert(x_store + 7 < dstimg.cols);
      assert(y / 3 * 2 < dstimg.rows);
      assert(y / 3 * 2 + 1 < dstimg.rows);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(
              &(dstimg.at<uint16_t>(y / 3 * 2, x_store))), store0);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(
              &(dstimg.at<uint16_t>(y / 3 * 2 + 1, x_store))), store2);

      x_store += 8;

      if (end)
        break;
      if (x + 12 >= colsMax) {
        x = colsMax - 12;
        x_store = dstimg.cols - 8;
        end = true;
      }
    }
  }
#endif
}

void Twothirdsample8(const agast::Mat& srcimg, agast::Mat& dstimg) {
  // Take care with border...
  const uint16_t leftoverCols = ((srcimg.cols / 3) * 3) % 15;

  // Make sure the destination image is of the right size:
  CHECK_EQ((srcimg.cols / 3) * 2, dstimg.cols);
  CHECK_EQ((srcimg.rows / 3) * 2, dstimg.rows);

  // Data pointers:
  unsigned char* p1 = srcimg.data;
  unsigned char* p2 = p1 + srcimg.cols;
  unsigned char* p3 = p2 + srcimg.cols;
  unsigned char* p_dest1 = dstimg.data;
  unsigned char* p_dest2 = p_dest1 + dstimg.cols;
  unsigned char* p_end = p1 + (srcimg.cols * srcimg.rows);

  unsigned int row = 0;
  unsigned int row_dest = 0;
  int hsize = srcimg.cols / 15;

#ifdef __ARM_NEON
  // masks:
    const uint8_t tmpmask1[16] = {1, 0x80, 4, 0x80, 7, 0x80, 10, 0x80, 13, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
    const uint8_t tmpmask2[16] = {0x80, 1, 0x80, 4, 0x80, 7, 0x80, 10, 0x80, 13,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
    const uint8_t tmpmask[16] = {0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 0x80, 0x80,
      0x80, 0x80};
    const uint8_t tmpstore_mask[16] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        // Lacking the masked storing intrinsics in NEON.
      0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0};
    uint8x16_t store_mask = vld1q_u8(&tmpstore_mask[0]);

    while (p3 < p_end) {
      for (int i = 0; i < hsize; ++i) {
        // Load three rows:
        uint8x16_t first = vld1q_u8(reinterpret_cast<const uint8_t*>(p1));
        uint8x16_t second = vld1q_u8(reinterpret_cast<const uint8_t*>(p2));
        uint8x16_t third = vld1q_u8(reinterpret_cast<const uint8_t*>(p3));

        // Upper row:
        uint8_t upper[16];
        vst1q_u8(&upper[0], vrhaddq_u8(vrhaddq_u8(first, second), first));
        uint8_t shufuppermask1[16];
        uint8_t shufuppermask2[16];
        for (int shuffleidx = 0; shuffleidx < 16; ++shuffleidx) {
          shufuppermask1[shuffleidx] =
          (tmpmask1[shuffleidx] & 0x80) ?
          0 : upper[tmpmask1[shuffleidx] & 0x0F];
          shufuppermask2[shuffleidx] =
          (tmpmask2[shuffleidx] & 0x80) ?
          0 : upper[tmpmask2[shuffleidx] & 0x0F];
        }
        uint8x16_t temp1_upper = vorrq_u8(vld1q_u8(&shufuppermask1[0]),
            vld1q_u8(&shufuppermask2[0]));

        uint8_t temp2_upper_array[16];
        for (int shuffleidx = 0; shuffleidx < 16; ++shuffleidx) {
          temp2_upper_array[shuffleidx] =
          (tmpmask[shuffleidx] & 0x80) ?
          0 : upper[tmpmask[shuffleidx] & 0x0F];
        }
        uint8x16_t temp2_upper = vld1q_u8(&temp2_upper_array[0]);
        uint8x16_t result_upper = vrhaddq_u8(vrhaddq_u8(temp2_upper, temp1_upper),
            temp2_upper);

        // Lower row:

        uint8_t lower[16];
        vst1q_u8(&lower[0], vrhaddq_u8(vrhaddq_u8(third, second), third));
        uint8_t shuflowermask1[16];
        uint8_t shuflowermask2[16];
        uint8_t temp2_lower_array[16];
        for (int shuffleidx = 0; shuffleidx < 16; ++shuffleidx) {
          shuflowermask1[shuffleidx] =
          (tmpmask1[shuffleidx] & 0x80) ?
          0 : lower[tmpmask1[shuffleidx] & 0x0F];
          shuflowermask2[shuffleidx] =
          (tmpmask2[shuffleidx] & 0x80) ?
          0 : lower[tmpmask2[shuffleidx] & 0x0F];
          temp2_lower_array[shuffleidx] =
          (tmpmask[shuffleidx] & 0x80) ?
          0 : lower[tmpmask[shuffleidx] & 0x0F];
        }
        uint8x16_t temp1_lower = vorrq_u8(vld1q_u8(&shuflowermask1[0]),
            vld1q_u8(&shuflowermask2[0]));
        uint8x16_t temp2_lower = vld1q_u8(&temp2_lower_array[0]);

        uint8x16_t result_lower = vrhaddq_u8(vrhaddq_u8(temp2_lower, temp1_lower),
            temp2_lower);

        // Store:
        if (i * 10 + 16 > dstimg.cols) {
          // Mask necessary data to store and mask with data already existing:
          uint8x16_t uppermasked = vorrq_u8(vandq_u8(result_upper, store_mask),
              vld1q_u8(p_dest1));
          uint8x16_t lowermasked = vorrq_u8(vandq_u8(result_lower, store_mask),
              vld1q_u8(p_dest2));

          vst1q_u8(p_dest1, uppermasked);
          vst1q_u8(p_dest2, lowermasked);
        } else {
          vst1q_u8(reinterpret_cast<uint8_t*>(p_dest1), result_upper);
          vst1q_u8(reinterpret_cast<uint8_t*>(p_dest2), result_lower);
        }

        // Shift pointers:
        p1 += 15;
        p2 += 15;
        p3 += 15;
        p_dest1 += 10;
        p_dest2 += 10;
      }

      // Fill the remainder:
      for (unsigned int j = 0; j < leftoverCols; j += 3) {
        const uint16_t A1 = *(p1++);
        const uint16_t A2 = *(p1++);
        const uint16_t A3 = *(p1++);
        const uint16_t B1 = *(p2++);
        const uint16_t B2 = *(p2++);
        const uint16_t B3 = *(p2++);
        const uint16_t C1 = *(p3++);
        const uint16_t C2 = *(p3++);
        const uint16_t C3 = *(p3++);

        *(p_dest1++) = static_cast<unsigned char>(
            ((4 * A1 + 2 * (A2 + B1 + 1) + B2 + 1) / 9) & 0x00FF);
        *(p_dest1++) = static_cast<unsigned char>(
            ((4 * A3 + 2 * (A2 + B3 + 1) + B2 + 1) / 9) & 0x00FF);
        *(p_dest2++) = static_cast<unsigned char>(
            ((4 * C1 + 2 * (C2 + B1 + 1) + B2 + 1) / 9) & 0x00FF);
        *(p_dest2++) = static_cast<unsigned char>(
            ((4 * C3 + 2 * (C2 + B3 + 1) + B2 + 1) / 9) & 0x00FF);
      }

      // Increment row counter:
      row += 3;
      row_dest += 2;

      // Reset pointers:
      p1 = srcimg.data + row * srcimg.cols;
      p2 = p1 + srcimg.cols;
      p3 = p2 + srcimg.cols;
      p_dest1 = dstimg.data + row_dest * dstimg.cols;
      p_dest2 = p_dest1 + dstimg.cols;
    }
#else
  // Masks:
  __m128i mask1 = _mm_set_epi8(-128, -128, -128, -128, -128, -128,
                                        -128, 13, -128, 10, -128, 7, -128, 4,
                                        -128, 1);
  __m128i mask2 = _mm_set_epi8(-128, -128, -128, -128, -128, -128,
                                        13, -128, 10, -128, 7, -128, 4, -128,
                                        1, -128);
  __m128i mask = _mm_set_epi8(-128, -128, -128, -128, -128, -128, 14,
                                       12, 11, 9, 8, 6, 5, 3, 2, 0);
  __m128i store_mask = _mm_set_epi8(0, 0, 0, 0, 0, 0, -128, -128, -128,
                                             -128, -128, -128, -128, -128, -128,
                                             -128);

  while (p3 < p_end) {
    for (int i = 0; i < hsize; ++i) {
      // Load three rows
      __m128i first = _mm_loadu_si128(reinterpret_cast<__m128i *>(p1));
      __m128i second = _mm_loadu_si128(reinterpret_cast<__m128i *>(p2));
      __m128i third = _mm_loadu_si128(reinterpret_cast<__m128i *>(p3));

      // Upper row:
      __m128i upper = _mm_avg_epu8(_mm_avg_epu8(first, second), first);
      __m128i temp1_upper = _mm_or_si128(_mm_shuffle_epi8(upper, mask1),
                                         _mm_shuffle_epi8(upper, mask2));
      __m128i temp2_upper = _mm_shuffle_epi8(upper, mask);
      __m128i result_upper = _mm_avg_epu8(
          _mm_avg_epu8(temp2_upper, temp1_upper), temp2_upper);

      // Lower row:
      __m128i lower = _mm_avg_epu8(_mm_avg_epu8(third, second), third);
      __m128i temp1_lower = _mm_or_si128(_mm_shuffle_epi8(lower, mask1),
                                         _mm_shuffle_epi8(lower, mask2));
      __m128i temp2_lower = _mm_shuffle_epi8(lower, mask);
      __m128i result_lower = _mm_avg_epu8(
          _mm_avg_epu8(temp2_lower, temp1_lower), temp2_lower);

      // Store:
      if (i * 10 + 16 > dstimg.cols) {
        _mm_maskmoveu_si128(result_upper, store_mask,
                            reinterpret_cast<char*>(p_dest1));
        _mm_maskmoveu_si128(result_lower, store_mask,
                            reinterpret_cast<char*>(p_dest2));
      } else {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(p_dest1), result_upper);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(p_dest2), result_lower);
      }

      // Shift pointers:
      p1 += 15;
      p2 += 15;
      p3 += 15;
      p_dest1 += 10;
      p_dest2 += 10;
    }

    // Fill the remainder:
    for (unsigned int j = 0; j < leftoverCols; j += 3) {
      const uint16_t A1 = *(p1++);
      const uint16_t A2 = *(p1++);
      const uint16_t A3 = *(p1++);
      const uint16_t B1 = *(p2++);
      const uint16_t B2 = *(p2++);
      const uint16_t B3 = *(p2++);
      const uint16_t C1 = *(p3++);
      const uint16_t C2 = *(p3++);
      const uint16_t C3 = *(p3++);

      *(p_dest1++) = static_cast<unsigned char>(((4 * A1 + 2 * (A2 + B1 + 1)
          + B2 + 1) / 9) & 0x00FF);
      *(p_dest1++) = static_cast<unsigned char>(((4 * A3 + 2 * (A2 + B3 + 1)
          + B2 + 1) / 9) & 0x00FF);
      *(p_dest2++) = static_cast<unsigned char>(((4 * C1 + 2 * (C2 + B1 + 1)
          + B2 + 1) / 9) & 0x00FF);
      *(p_dest2++) = static_cast<unsigned char>(((4 * C3 + 2 * (C2 + B3 + 1)
          + B2 + 1) / 9) & 0x00FF);
    }

    // Increment row counter:
    row += 3;
    row_dest += 2;

    // Reset pointers
    p1 = srcimg.data + row * srcimg.cols;
    p2 = p1 + srcimg.cols;
    p3 = p2 + srcimg.cols;
    p_dest1 = dstimg.data + row_dest * dstimg.cols;
    p_dest2 = p_dest1 + dstimg.cols;
  }
#endif  // __ARM_NEON
}
}  // namespace brisk
