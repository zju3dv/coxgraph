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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * frame-absolute-pose-sac-problem.hpp
 * @brief Header for the FrameAbsolutePoseSacProblem class
 * @author: Marco Karrer
 * Created on: Aug 16, 2018
 */

#pragma once

// The opengv includes
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/types.hpp>

#include "opengv/absolute_pose/frame-noncentral-absolute-adapter.hpp"

namespace opengv {

namespace sac_problems {

namespace absolute_pose {

/// @brief Provides functions for fitting an absolute-pose model to a set of
///        bearing-vector to point correspondences, using different algorithms
///        (central and non-central ones). Used in a sample-consenus paradigm
///        for rejecting outlier correspondences.
class FrameAbsolutePoseSacProblem : public AbsolutePoseSacProblem {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef AbsolutePoseSacProblem base_t;

  /** The type of adapter that is expected by the methods */
  using base_t::adapter_t;
  /** The possible algorithms for solving this problem */
  using base_t::algorithm_t;
  /** The model we are trying to fit (transformation) */
  using base_t::model_t;

  /// @brief Constructor.
  /// @param adapter Visitor holding bearing vectors, world points, etc.
  /// @param algorithm The algorithm we want to use.
  /// @warning Only absolute_pose::FrameNoncentralAbsoluteAdapter supported.
  FrameAbsolutePoseSacProblem(adapter_t& adapter, algorithm_t algorithm)
      : base_t(adapter, algorithm),
        adapterDerived_(*static_cast<
                        opengv::absolute_pose::FrameNoncentralAbsoluteAdapter*>(
            &_adapter)) {}

  /// @brief Constructor.
  /// @param adapter Visitor holding bearing vectors, world points, etc.
  /// @param algorithm The algorithm we want to use.
  /// @param indices A vector of indices to be used from all available
  ///                    correspondences.
  /// @warning Only absolute_pose::FrameNoncentralAbsoluteAdapter supported.
  FrameAbsolutePoseSacProblem(adapter_t& adapter, algorithm_t algorithm,
                              const std::vector<int>& indices)
      : base_t(adapter, algorithm, indices),
        adapterDerived_(*static_cast<
                        opengv::absolute_pose::FrameNoncentralAbsoluteAdapter*>(
            &_adapter)) {}

  virtual ~FrameAbsolutePoseSacProblem() {}

  /// @brief Compute the distances of all samples whith respect to given model
  ///        coefficients.
  /// @param model The coefficients of the model hypothesis.
  /// @param indices The indices of the samples of which we compute distances.
  /// @param scores The resulting distances of the selected samples. Low
  ///                    distances mean a good fit.
  virtual void getSelectedDistancesToModel(const model_t& model,
                                           const std::vector<int>& indices,
                                           std::vector<double>& scores) const {
    // compute the reprojection error of all points
    // compute inverse transformation
    model_t inverseSolution;
    inverseSolution.block<3, 3>(0, 0) = model.block<3, 3>(0, 0).transpose();
    inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0) * model.col(3);

    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;

    for (size_t i = 0; i < indices.size(); i++) {
      // get point in homogeneous form
      p_hom.block<3, 1>(0, 0) = adapterDerived_.getPoint(indices[i]);

      // compute the reprojection (this is working for both central and
      // non-central case)
      point_t bodyReprojection = inverseSolution * p_hom;
      point_t reprojection =
          adapterDerived_.getCamRotation(indices[i]).transpose() *
          (bodyReprojection - adapterDerived_.getCamOffset(indices[i]));
      reprojection = reprojection / reprojection.norm();

      // compute the score
      point_t error =
          (reprojection - adapterDerived_.getBearingVector(indices[i]));
      double error_squared = error.transpose() * error;
      scores.push_back(error_squared /
                       adapterDerived_.getSigmaAngle(indices[i]));
    }
  }

 protected:
  /// The adapter holding the bearing, correspondences etc.
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter& adapterDerived_;
};

}  // namespace absolute_pose
}  // namespace sac_problems
}  // namespace opengv
