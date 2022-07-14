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
#include <cstring>
#include <functional>
#include <iostream>  // NOLINT
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "./image-io.h"

namespace brisk {

bool NumericStringCompare(const std::string& s1, const std::string& s2) {
  using namespace std;
  std::string::const_iterator it1 = s1.begin(), it2 = s2.begin();
  std::string::const_iterator rit1 = s1.end(), rit2 = s2.end();
  //find beginning of number
  --rit1;
  --rit2;
  while (!std::isdigit(*rit1)) {
    --rit1;
  }
  it1 = rit1;
  while (std::isdigit(*it1)) {
    --it1;
  }
  ++it1;
  while (!std::isdigit(*rit2)) {
    --rit2;
  }
  it2 = rit2;
  while (std::isdigit(*it2)) {
    --it2;
  }
  ++it2;

  if (rit1 - it1 == rit2 - it2) {
    while (it1 != rit1 && it2 != rit2) {
      if (*it1 > *it2) {
        return true;
      } else if (*it1 < *it2) {
        return false;
      }
      ++it1;
      ++it2;
    }
    if (*it1 > *it2) {
      return true;
    } else {
      return false;
    }
  } else if (rit1 - it1 > rit2 - it2) {
    return true;
  } else {
    return false;
  }
}

int Getfilelists(const std::vector<std::string>& initialPaths,
                 bool sortlexical,
                 const std::string& extension_filter,
                 std::vector<std::string>* imagepaths) {
  std::string initialPath;
  DIR * d;
  struct dirent *dir;
  for (size_t diridx = 0; diridx < initialPaths.size(); ++diridx) {
    initialPath = initialPaths.at(diridx);
    d = opendir(initialPath.c_str());
    if (d == NULL) {
      throw std::logic_error(initialPath + " results in d == NULL");
      return 1;
    }
    int i = 0;
    while ((dir = readdir(d))) {
      if (strcmp(dir->d_name, ".") == 0 ||
          strcmp(dir->d_name, "..") == 0) {
        continue;
      }
      if (!std::regex_match(dir->d_name,
                            std::regex("(.*)(" + extension_filter + ")"))) {
        std::cout << "Skipped file: " << dir->d_name << std::endl;
        continue;
      }
      if (dir == NULL)
        break;

      i++;
      imagepaths->push_back(initialPath + dir->d_name);
    }
  }
  if (sortlexical) {
    sort(imagepaths->begin(), imagepaths->end());  //normal lexical sort
  } else {
    sort(
         imagepaths->begin(),
         imagepaths->end(),
         std::bind(&NumericStringCompare, std::placeholders::_2,
                   std::placeholders::_1));  //sorts strictly by the number in the file name
  }
  return 0;
}
}  // namespace brisk
