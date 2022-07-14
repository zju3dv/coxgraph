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
 * common.h
 * @brief Header file for commonly used operations.
 * @author: Marco Karrer
 * Created on: Mar 19, 2018
 */

#pragma once

#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

/// \brief robopt The main namespace of this package.
namespace robopt {

namespace common {

namespace weighting {

inline double cauchyWeight(const double resid, const double c);

} // namespace weighting

namespace quaternion {

inline Eigen::Matrix3d Gamma(const Eigen::Vector3d& phi);

inline Eigen::Quaterniond ExpMap(const Eigen::Vector3d& theta);
inline Eigen::Vector3d LogMap(const Eigen::Quaterniond& q);

// Implements the boxplus operator
// q_res = q boxplus delta
inline void Plus(
  const Eigen::Ref<const Eigen::Vector4d>& q,
  const Eigen::Ref<const Eigen::Vector3d>& delta,
  Eigen::Quaterniond* q_res);
inline void Plus(
  const Eigen::Quaterniond& q,
  const Eigen::Ref<const Eigen::Vector3d>& delta,
  Eigen::Quaterniond* q_res);


// Implements the boxminus operator
// p_minus_q = p boxminus q
inline void Minus(
    const Eigen::Quaterniond& p, const Eigen::Quaterniond& q,
    Eigen::Vector3d* p_minus_q);

// Implements small angle approximation
// q = [1.0, 0.5*theta]
inline Eigen::Quaterniond DeltaQ(
    const Eigen::Vector3d& theta);


// Plus matrix of a quaternion, i.e. q_AB*q_BC = plus(q_AB)*q_BC.coeffs().
// input: q_AB.coeffs() A Quaternion.
inline Eigen::Matrix4d PlusMat(const Eigen::Vector4d& q);

inline Eigen::Matrix4d PlusMat(const Eigen::Quaterniond & q_AB);

// Oplus matrix of a quaternion, i.e. q_AB*q_BC = oplus(q_BC)*q_AB.coeffs().
// input q_BC.coeffs() A Quaternion.
inline Eigen::Matrix4d OPlusMat(const Eigen::Vector4d & q);

inline Eigen::Matrix4d OPlusMat(const Eigen::Quaterniond & q_BC);

} // namepsace quaternion

inline Eigen::Matrix3d skew(
    const Eigen::Vector3d& vector);

namespace yaw {

inline Eigen::Quaterniond ExpMap(const double yaw);
inline double LogMap(const Eigen::Quaterniond& q);
inline Eigen::Matrix<double, 1, 4> LogMapJacobian(const Eigen::Quaterniond& q);

// Implements the boxplus operator
// q_res = q boxplus delta
inline void Plus(
  const Eigen::Ref<const Eigen::Vector4d>& q,
  const double& delta,
  Eigen::Quaterniond* q_res);
inline void Plus(
  const Eigen::Quaterniond& q,
  const double& delta,
  Eigen::Quaterniond* q_res);


// Implements the boxminus operator
// p_minus_q = p boxminus q
inline void Minus(
    const Eigen::Quaterniond& p, const Eigen::Quaterniond& q,
    double* p_minus_q);

// Transform the yaw value into the range [-180°, 180°]
inline double normalizeYaw(const double yaw);

} // namespace yaw

} // namespace common

} // namespace robopt

#include "./common-inl.h"
