/*
* Copyright (c) 2018, Vision for Robotics Lab
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the Vision for Robotics Lab, ETH Zurich nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

/*
 * definitions.h
 * @brief Header file for various useful definitions.
 * @author: Marco Karrer
 * Created on: Mar 19, 2018
 */

#pragma once

#include <eigen3/Eigen/Dense>
#include <Eigen/StdVector>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace defs {

namespace visual {

static const int kResidualSize = 2;

static const int kOrientationBlockSize = 4;
static const int kUnit3BlockSize = 4;
static const int kInverseDepthBlockSize = 1;
static const int kPositionBlockSize = 3;
static const int kPoseBlockSize = 7;

// Specify in which direction the relative projection is perfomed.
// kNormal: The optimized relative pose is directly used to transform points.
// kInverse: The inversed relative pose is used to transform points.
enum RelativeProjectionType {kNormal, kInverse};

} // namespace visual

namespace pose {

// Specify in which coordinate frame the measurement is expressed
// kVisual: The error is expressed in the camera frame
// kImu: The error is expressed in the imu frame
enum PoseErrorType {kVisual, kImu};

static const int kResidualSize = 6;
static const int kResidualSizeYaw = 4;
static const int kResidualSizePosition = 3;

static const int kOrientationBlockSize = 4;
static const int kPositionBlockSize = 3;
static const int kPoseBlockSize = 7;
static const int kSpeedBiasBlockSize = 9;

// Order for the state in the 15x1 vector
enum StateOrder {
  kRotation = 0,
  kPosition = 3,
  kVelocity = 6,
  kBiasA = 9,
  kBiasG = 12
};

} // namespace pose

} // namespace defs

} // namespace robopt
