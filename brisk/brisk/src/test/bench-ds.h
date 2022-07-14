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

#ifndef TEST_BENCH_DS_H_
#define TEST_BENCH_DS_H_

#include <fstream>  // NOLINT
#include <iostream>  // NOLINT
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <brisk/brisk.h>
#include <brisk/internal/hamming.h>
#include <agast/wrap-opencv.h>

#include "./serialization.h"

namespace brisk {

typedef unsigned char imagedata_T;
struct Blob;
struct DatasetEntry;

#define EXPECTSAMETHROW(THIS, OTHER, MEMBER) \
    do { if (THIS.MEMBER != OTHER.MEMBER) { \
      std::stringstream ss; \
      ss<<  "Failed on " << #MEMBER << ": "<< THIS.MEMBER \
      << " other " << OTHER.MEMBER << " at " << __PRETTY_FUNCTION__ \
      << " Line: " << __LINE__ << std::endl; \
      CHECK(false) << ss.str(); \
      return false;\
    } } while (0);

#define EXPECTSAMETHROWACCESSOR(THIS, OTHER, ACCESSOR, KEYPOINTIDX) \
    do { if (ACCESSOR(THIS) != ACCESSOR(OTHER)) { \
      std::stringstream ss; \
      ss <<  "For Keypoint " << KEYPOINTIDX << " failed on " \
      <<  "Failed on " << #ACCESSOR << ": "<< ACCESSOR(THIS) \
      << " other " << ACCESSOR(OTHER) << " at " << __PRETTY_FUNCTION__ \
      << " Line: " << __LINE__ << std::endl; \
      CHECK(false) << ss.str(); \
      return false;\
    } } while (0);

#define CHECKCVKEYPOINTMEMBERSAME(THIS, OTHER, MEMBER, KEYPOINTIDX) \
    do { if (THIS.MEMBER != OTHER.MEMBER) { \
      std::stringstream ss; \
      ss <<  "For Keypoint " << KEYPOINTIDX << " failed on " \
      << #MEMBER << ": "<< THIS.MEMBER << \
      " other " << OTHER.MEMBER << \
      " at " << __PRETTY_FUNCTION__ << " Line: " << __LINE__ << std::endl; \
      ss << "this / other: "\
      << std::endl << "pt.x:\t" << agast::KeyPointX(THIS) << "\t"\
      << agast::KeyPointX(OTHER) << std::endl\
      << std::endl << "pt.y:\t" << agast::KeyPointY(THIS) << "\t"\
      << agast::KeyPointY(OTHER) << std::endl << std::endl \
      << std::endl << "octave:\t" << agast::KeyPointOctave(THIS) \
      << "\t" << agast::KeyPointOctave(OTHER) \
      << std::endl << "response:\t" << agast::KeyPointResponse(THIS) \
      << "\t" << agast::KeyPointResponse(OTHER) \
      << std::endl << "size:\t" << THIS.size << "\t" << OTHER.size \
      << std::endl; \
      CHECK(false) << ss.str(); \
      return false;\
    } } while (0);

#define CHECKCVKEYPOINTANGLESAME(THIS, OTHER, MEMBER, KEYPOINTIDX) \
    do { if ((std::abs(THIS.MEMBER - OTHER.MEMBER) > 180 ? \
      (360 - std::abs(THIS.MEMBER-OTHER.MEMBER)) : \
      std::abs(THIS.MEMBER - OTHER.MEMBER) ) > 0.01) { \
      std::stringstream ss; \
      ss <<  "For Keypoint " << KEYPOINTIDX \
      << " failed on angle" << ": "<< THIS.MEMBER\
      << " other " << OTHER.MEMBER \
      << " at " << __PRETTY_FUNCTION__ << " Line: " << __LINE__ << std::endl; \
      ss << "this / other: "\
      << std::endl << "pt.x:\t" << agast::KeyPointX(THIS) << "\t"\
      << agast::KeyPointX(OTHER) << std::endl\
      << std::endl << "pt.y:\t" << agast::KeyPointY(THIS) << "\t"\
      << agast::KeyPointY(OTHER) << std::endl << std::endl \
      << std::endl << "octave:\t" << agast::KeyPointOctave(THIS) \
      << "\t" << agast::KeyPointOctave(OTHER) \
      << std::endl << "response:\t" << agast::KeyPointResponse(THIS) \
      << "\t" << agast::KeyPointResponse(OTHER) \
      << std::endl << "size:\t" << THIS.size << "\t" << OTHER.size \
      << std::endl;\
      CHECK(false) << ss.str(); \
      return false;\
    } } while (0);

#ifdef __ARM_NEON
  std::string DescriptorToString(const uint8x16_t* d, int num128Words);
#else
  std::string DescriptorToString(const __m128i * d, int num128Words);
#endif

struct Blob {
  friend void Serialize(const Blob& value, std::ofstream* out);
  friend void DeSerialize(Blob* value, std::ifstream* in);
 private:
  std::unique_ptr<unsigned char[]> current_data_;
  std::unique_ptr<unsigned char[]> verification_data_;
  uint32_t size_;
 public:
  Blob() {
    size_ = 0;
  }
  Blob(const Blob& other) {
    size_ = other.size_;
    if (size_) {
      verification_data_.reset(new unsigned char[size_]);
      memcpy(verification_data_.get(), other.verification_data_.get(), size_);
      if (other.current_data_) {
        current_data_.reset(new unsigned char[size_]);
        memcpy(current_data_.get(), other.current_data_.get(), size_);
      }
    }
  }

  Blob& operator=(const Blob& other) {
    if( this != &other) {
      size_ = other.size_;
      if (size_) {
        verification_data_.reset(new unsigned char[size_]);
        memcpy(verification_data_.get(), other.verification_data_.get(), size_);
        if (other.current_data_) {
          current_data_.reset(new unsigned char[size_]);
          memcpy(current_data_.get(), other.current_data_.get(), size_);
        }
      }
    }
    return *this;
  }


  uint32_t Size() const {
    return size_;
  }

  int CheckCurrentDataSameAsVerificationData() const {
    if (!size_) {
      return 0;
    }
    // If we don't have data to compare, we assume the user has not set it.
    if (!current_data_) {
      std::cout
          << "You asked me to verify userdata, but the current_data is not set"
          << std::endl;
      return 0;
    }
    return memcmp(current_data_.get(), verification_data_.get(), size_);
  }

  bool HasverificationData() {
    return static_cast<bool>(verification_data_);
  }

  void SetVerificationData(const unsigned char* data, uint32_t size) {
    size_ = size;
    verification_data_.reset(new unsigned char[size_]);
    memcpy(verification_data_.get(), data, size_);
  }

  void SetCurrentData(const unsigned char* data, uint32_t size) {
    if (size != size_) {
      CHECK(false)
      << "You set the current data to a different length than "
      "the verification data. This will fail the verification."
      "Use both times the same lenght.";
    }
    current_data_.reset(new unsigned char[size]);
    memcpy(current_data_.get(), data, size);
  }

  const unsigned char* VerificationData() {
    if (verification_data_) {
      return verification_data_.get();
    } else {
      return NULL;
    }
  }

  const unsigned char* CurrentData() {
    if (current_data_) {
      return current_data_.get();
    } else {
      return NULL;
    }
  }
};

struct DatasetEntry {
  friend void DeSerialize(DatasetEntry* value, std::ifstream* in);
  friend void Serialize(const DatasetEntry& value, std::ofstream* out);
 private:
  std::map<std::string, Blob> userdata_;
  std::string path_;
  agast::Mat imgGray_;
  std::vector<agast::KeyPoint> keypoints_;
  agast::Mat descriptors_;

 public:
  DatasetEntry() = default;

  std::string Shortname() {
    int idx = path_.find_last_of("/\\") + 1;
    return path_.substr(idx, path_.length() - idx);
  }

  const std::string& GetPath() const {
    return path_;
  }

  const agast::Mat& GetImage() const {
    return imgGray_;
  }

  const std::vector<agast::KeyPoint>& GetKeyPoints() const {
    return keypoints_;
  }

  const agast::Mat& GetDescriptors() const {
    return descriptors_;
  }

  std::string* GetPathMutable() {
    return &path_;
  }

  agast::Mat* GetImgMutable() {
    return &imgGray_;
  }

  std::vector<agast::KeyPoint>* GetKeyPointsMutable() {
    return &keypoints_;
  }

  agast::Mat* GetDescriptorsMutable() {
    return &descriptors_;
  }

  /*
   * especially do a deep copy of the agast::Mats
   */
  DatasetEntry(const DatasetEntry& other) {
    path_ = other.path_;
    imgGray_ = other.imgGray_.clone();
    keypoints_ = other.keypoints_;
    descriptors_ = other.descriptors_.clone();
    userdata_ = other.userdata_;
  }

  std::string Path() const {
    return path_;
  }

  // Returns a blob of data belonging to this key.
  Blob& GetBlob(std::string key) {
    return userdata_[key];
  }

  std::string ListBlobs() const {
    std::stringstream ss;
    ss << "Userdata:" << std::endl;
    int i = 0;
    for (std::map<std::string, Blob>::const_iterator it = userdata_.begin(),
        end = userdata_.end(); it != end; ++it, ++i) {
      ss << "\t\t" << i << ": " << "" "" << it->first << "" " size: "
          << it->second.Size() << std::endl;
    }
    return ss.str();
  }

  bool operator==(const DatasetEntry& other) const {
    EXPECTSAMETHROW((*this), other, path_);

    // CHECK IMAGE.
    bool doImageVerification = true;  // Not really necessary.
    if (doImageVerification) {
      // Check rows.
      if (this->imgGray_.rows != other.imgGray_.rows) {
        EXPECTSAMETHROW((*this), other, imgGray_.rows);
        return false;
      }
      // Check cols.
      if (this->imgGray_.cols != other.imgGray_.cols) {
        EXPECTSAMETHROW((*this), other, imgGray_.cols);
        return false;
      }
      // Check type.
      if (this->imgGray_.type() != other.imgGray_.type()) {
        EXPECTSAMETHROW((*this), other, imgGray_.type());
        return false;
      }
      if(!this->imgGray_.data && other.imgGray_.data) {
        return false;
      }
      if (this->imgGray_.data && !other.imgGray_.data) {
        return false;
      }
      // Check pixel by pixel.
      for (int i = 0, size = this->imgGray_.rows * this->imgGray_.cols;
          i < size; ++i) {
        if (this->imgGray_.data[i] != other.imgGray_.data[i]) {
          std::stringstream ss;
          ss << "Failed on imgGray_.data at [" << i % this->imgGray_.cols
              << ", " << i / this->imgGray_.cols << "] "
              << static_cast<int>(this->imgGray_.data[i]) << " other "
              << static_cast<int>(other.imgGray_.data[i]) << " at "
              << __PRETTY_FUNCTION__ << " Line: " << __LINE__ << std::endl;
          CHECK(false) << ss.str();
          return false;
        }
      }
    }

    // CHECK KEYPOINTS.
    if (this->keypoints_.size() != other.keypoints_.size()) {
      EXPECTSAMETHROW((*this), other, keypoints_.size());
      return false;
    }

    // TODO(slynen): we might want to sort the keypoints and descriptors by
    // location to allow detection and description to be done with blocking type
    // optimizations.
    int kpidx = 0;
    for (std::vector<agast::KeyPoint>::const_iterator it_this = this->keypoints_
        .begin(), it_other = other.keypoints_.begin(), end_this = this
        ->keypoints_.end(), end_other = other.keypoints_.end();
        it_this != end_this && it_other != end_other;
        ++it_this, ++it_other, ++kpidx) {
#if HAVE_OPENCV
      CHECKCVKEYPOINTMEMBERSAME((*it_this), (*it_other), class_id, kpidx);
#endif
      EXPECTSAMETHROWACCESSOR((*it_this), (*it_other), agast::KeyPointX, kpidx);
      EXPECTSAMETHROWACCESSOR((*it_this), (*it_other), agast::KeyPointY, kpidx);
      EXPECTSAMETHROWACCESSOR((*it_this), (*it_other), agast::KeyPointAngle, kpidx);
      EXPECTSAMETHROWACCESSOR((*it_this), (*it_other), agast::KeyPointOctave, kpidx);
      EXPECTSAMETHROWACCESSOR((*it_this), (*it_other), agast::KeyPointResponse, kpidx);
      EXPECTSAMETHROWACCESSOR((*it_this), (*it_other), agast::KeyPointSize, kpidx);
    }

    // Check descriptors.
    if (this->descriptors_.rows != other.descriptors_.rows) {
      EXPECTSAMETHROW((*this), other, descriptors_.rows);
      return false;
    }
    if (this->descriptors_.cols != other.descriptors_.cols) {
      EXPECTSAMETHROW((*this), other, descriptors_.cols);
      return false;
    }

    uint32_t hammdisttolerance = 5;
    int numberof128Blocks = other.descriptors_.step * 8 / 128;
    for (int rowidx = 0; rowidx < this->descriptors_.rows; ++rowidx) {
#ifdef __ARM_NEON
      const uint8x16_t* d1 = reinterpret_cast<const uint8x16_t *>(
          this->descriptors_.data + this->descriptors_.step * rowidx);
      const uint8x16_t* d2 = reinterpret_cast<const uint8x16_t *>(
          other.descriptors_.data + other.descriptors_.step * rowidx);
      uint32_t hammdist = brisk::Hamming::NEONPopcntofXORed(
          d1, d2, numberof128Blocks);
#else
      const __m128i* d1 = reinterpret_cast<const __m128i *>(
          this->descriptors_.data + this->descriptors_.step * rowidx);
      const __m128i* d2 = reinterpret_cast<const __m128i *>(
          other.descriptors_.data + other.descriptors_.step * rowidx);
      uint32_t hammdist = brisk::Hamming::SSSE3PopcntofXORed(
          d1, d2, numberof128Blocks);
#endif
      if (hammdist > hammdisttolerance) {
        std::cout << "Failed on descriptor " << rowidx << ": Hammdist "
        << hammdist << " " << DescriptorToString(d1, numberof128Blocks) <<
        " other " << DescriptorToString(d2, numberof128Blocks) << std::endl;
        return false;
      }
    }

    // Check user data.
    for (std::map<std::string, Blob>::const_iterator it = userdata_.begin(),
        end = userdata_.end(); it != end; ++it) {
      int diffbytes = it->second.CheckCurrentDataSameAsVerificationData();
      if (diffbytes) {
        std::stringstream ss;
        ss << "For userdata " << it->first << " failed with " << diffbytes
            << " bytes difference. At " << __PRETTY_FUNCTION__ << " Line: "
            << __LINE__ << std::endl;
        CHECK(false)
        << ss.str();
      }
    }
    return true;
  }

  bool operator!=(const DatasetEntry& other) const {
    return !operator==(other);
  }

  // Echo some information about this entry.
  std::string print() const {
    std::stringstream ss;
    ss << "\t" << path_ << " [" << imgGray_.cols << "x" << imgGray_.rows << "]"
        << std::endl;
    if (!keypoints_.empty()) {
      ss << "\t Keypoints: " << keypoints_.size() << " Descriptors: "
          << descriptors_.rows << std::endl;
    }
    ss << "\t" << ListBlobs();
    return ss.str();
  }

  // Remove processing results so we can re-run the pipeline on this image.
  void clear_processed_data(bool clearDescriptors, bool clearKeypoints) {
    if (clearDescriptors) {
      descriptors_ = agast::Mat::zeros(0, 0, CV_8U);
    }
    if (clearKeypoints) {
      keypoints_.clear();
    }
  }

  // Get the images from the path and convert to grayscale.
  void readImage(const std::string& path) {
    path_ = path;
    imgGray_ = cv::imread(path_, cv::IMREAD_GRAYSCALE);
    std::cout << "Done reading image: " << imgGray_.rows << "x" <<
        imgGray_.cols << std::endl;
  }

  // Set the static image name to the current image.
  void setThisAsCurrentEntry() {
    current_entry = this;
  }

  // Return the name of the image currently processed.
  static DatasetEntry* getCurrentEntry() {
    return current_entry;
  }

 private:
  // A global tag which image is currently being processed.
  static DatasetEntry* current_entry;
};

void Serialize(const Blob& value, std::ofstream* out);

void DeSerialize(Blob* value, std::ifstream* in);

void Serialize(const DatasetEntry& value, std::ofstream* out);

void DeSerialize(DatasetEntry* value, std::ifstream* in);

}  // namespace brisk

#endif  // TEST_BENCH_DS_H_
