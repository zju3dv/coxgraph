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
 * quaternion-yaw-local-param.cpp
 * @brief Implementation file for local parameterization of yaw angles using 
 *        Quaternions
 * @author: Marco Karrer
 * Created on: Aug 18, 2018
 */

#include <local-parameterization/quaternion-yaw-local-param.h>
#include <common/common.h>

namespace robopt {

namespace local_param {

bool QuaternionYawLocalParameterization::Plus(
    const double* x, const double* delta, double* x_plus_delta) const {
  Eigen::Map<Eigen::Quaterniond> q_res(x_plus_delta);
  Eigen::Map<const Eigen::Vector4d> q_curr(x);
  
  Eigen::Quaterniond q_tmp;
  common::yaw::Plus(q_curr, (*delta), &q_tmp);
  
  q_res = q_tmp;
  return true;
}

bool QuaternionYawLocalParameterization::ComputeJacobian(
    const double* x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 4, 1>> j(jacobian);
  const Eigen::Map<const Eigen::Quaterniond> quat_x(x);

  // Jacobian
  jacobian[0] = -quat_x.y() * 0.5;
  jacobian[1] =  quat_x.x() * 0.5;
  jacobian[2] =  quat_x.w() * 0.5;
  jacobian[3] = -quat_x.z() * 0.5;

  return true;
}

} // namespace local_param

} // namespace robopt
