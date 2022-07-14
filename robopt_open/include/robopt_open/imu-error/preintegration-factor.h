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
 * preintegration-factor.h
 * @brief Header file for the imu-preintegration factor.
 * @author: Marco Karrer
 * Created on: Jun 7, 2018
 */

#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <robopt_open/imu-error/preintegration-base.h>
#include <robopt_open/local-parameterization/quaternion-local-param.h>
#include <robopt_open/common/definitions.h>
#include <robopt_open/common/common.h>

/// \brief robopt Main namespace of this package
namespace robopt {

namespace imu {

class PreintegrationFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
  PreintegrationFactor() = delete;

  /// \brief Construct a preintegration error term.
  /// @param preint Pointer to the preintegration base building up the pre-
  ///        integrated measurement.
  PreintegrationFactor(PreintegrationBase* preint);

  virtual ~PreintegrationFactor() {};

  /// \brief The Evaluate function inherited from ceres.
  virtual bool Evaluate(const double * const *parameters,
      double *residuals, double **jacobians) const;

protected:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks.
  enum {
    kIdxPose1,
    kIdxSpeedBias1,
    kIdxPose2,
    kIdxSpeedBias2
  };

  // The representation for Jacobians computed by this object.
  typedef Eigen::Matrix<double, 15,
    defs::pose::kPoseBlockSize, Eigen::RowMajor> PoseJacobian;
  typedef Eigen::Matrix<double, 15,
    6, Eigen::RowMajor> PoseJacobianMin;
  typedef Eigen::Matrix<double, 15,
    defs::pose::kSpeedBiasBlockSize, Eigen::RowMajor> SpeedBiasJacobian;

  // Store the preintegration pointer.
  PreintegrationBase* pre_integration_;

};

} // namespace imu

} // namespace robopt
