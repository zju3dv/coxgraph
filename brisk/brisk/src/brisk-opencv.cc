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

#include <dirent.h>
#include <fcntl.h>
#include <fstream>  // NOLINT
#include <memory>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>

#include <agast/wrap-opencv.h>
#include <agast/glog.h>

#if !HAVE_OPENCV
namespace {
void GetNextUncommentedLine(std::ifstream& infile, std::string* input_line) {
  std::getline(infile, *input_line);
  while ((*input_line)[0] == '#') {
    std::getline(infile, *input_line);
  }
}
// O_DIRECT writing uses DMA and to achieve this needs the input memory to be
// memory aligned to multiples of 4096.  This will allocate properly aligned
// memory. See man 2 memalign for details.
#ifdef ANDROID
#define posix_memalign(a, b, c) (((*a) = memalign(b, c)) == NULL)
#endif
}  // namespace

namespace agast {
// Reads a pgm image from file.
agast::Mat imread(const std::string& filename) {
  std::ifstream infile;
  infile.open(filename.c_str());
  if (infile.fail()) {
    infile.close();
    return agast::Mat();
  }

  std::string input_line;
  // First line: version.
  GetNextUncommentedLine(infile, &input_line);
  // Second line: size.
  GetNextUncommentedLine(infile, &input_line);
  int cols, rows;
  {
    std::stringstream ss;
    ss << input_line;
    ss >> cols >> rows;
  }

  if (!cols || !rows) {
    return agast::Mat();
  }

  // Encoding
  GetNextUncommentedLine(infile, &input_line);

  // Create image and read in data.
  agast::Mat img(rows, cols, CV_8UC1);
  size_t size = rows * cols * img.elemSize();
  CHECK_NOTNULL(img.data);
  infile.read(reinterpret_cast<char*>(img.data), size);

  infile.close();
  return img;
}

bool MakePGMHeader(const agast::Mat& image, unsigned char** pgm_header) {
  // To use O_DIRECT to write to files, the memory size must be a multiple of
  // 4096 bytes and aligned to addresses multiples of 4096 at the source
  if (posix_memalign(reinterpret_cast<void **>(pgm_header), 4096, 4096)) {
    LOG(ERROR) << "Could not allocate memory for PGM header\n";
    return false;
  }
  memset(*pgm_header, '#', 4096);

  char head[32];
  int n = snprintf(head, sizeof(head), "P5\n%d %d\n######", image.cols,
                   image.rows);
  std::string head_string(head, n);
  memcpy(*pgm_header, head_string.c_str(), head_string.length());

  std::string foot_string;
  foot_string = "\n255\n";
  memcpy(*pgm_header + 4096 - 5, foot_string.c_str(), foot_string.length());

  return true;
}

ssize_t imwrite(const agast::Mat& image, const std::string& filepath) {
  ssize_t bytes_written = 0;

  int fd = open(filepath.c_str(), O_CREAT | O_RDWR | O_DIRECT,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
  if (fd < 0) {
    LOG(ERROR) << "Unable to open file " << filepath;
    return bytes_written;
  }

  unsigned char* pgm_header = nullptr;
  MakePGMHeader(image, &pgm_header);

  // The header is always written.
  bytes_written += write(fd, pgm_header, 4096);

  bytes_written += write(fd, image.data, image.cols * image.rows);
  if (pgm_header) {
    free (pgm_header);
    pgm_header = NULL;
  }
  close(fd);
  return bytes_written;
}

}  // namespace agast
#endif  // !HAVE_OPENCV
