#ifndef VOXGRAPH_BACKEND_CONSTRAINT_CONSTRAINT_COLLECTION_H_
#define VOXGRAPH_BACKEND_CONSTRAINT_CONSTRAINT_COLLECTION_H_

#include <list>
#include <vector>

#include "voxgraph/backend/constraint/absolute_pose_constraint.h"
#include "voxgraph/backend/constraint/registration_constraint.h"
#include "voxgraph/backend/constraint/relative_pose_constraint.h"

namespace voxgraph {
class ConstraintCollection {
 public:
  void addAbsolutePoseConstraint(const AbsolutePoseConstraint::Config& config) {
    absolute_pose_constraints_.emplace_back(newConstraintId(), config);
  }
  void addRelativePoseConstraint(const RelativePoseConstraint::Config& config) {
    relative_pose_constraints_.emplace_back(newConstraintId(), config);
  }
  void addRegistrationConstraint(const RegistrationConstraint::Config& config) {
    registration_constraints_.emplace_back(newConstraintId(), config);
  }
  void addSubmapRelativePoseConstraint(
      const RelativePoseConstraint::Config& config) {
    submap_relative_pose_constraints_.emplace_back(newConstraintId(), config);
  }
  void addForceRegistrationConstraint(
      const RegistrationConstraint::Config& config) {
    force_registration_constraints_.emplace_back(newConstraintId(), config);
  }

  void resetRegistrationConstraints() { registration_constraints_.clear(); }
  void resetSubmapRelativePoseConstraints() {
    submap_relative_pose_constraints_.clear();
  }
  void resetForceRegistrationConstraints() {
    force_registration_constraints_.clear();
  }

  void addConstraintsToProblem(const NodeCollection& node_collection,
                               ceres::Problem* problem_ptr,
                               bool exclude_registration_constraints = false);

  enum ConstraintType { RelPose = 0, SubmapRelPose };
  std::vector<ceres::ResidualBlockId> getResidualBlockIds(
      ConstraintType constraint_type) {
    std::vector<ceres::ResidualBlockId> res_block_ids;
    switch (constraint_type) {
      case ConstraintType::SubmapRelPose:
        for (auto& constraint : submap_relative_pose_constraints_) {
          res_block_ids.emplace_back(constraint.getResidualBlockId());
        }
        break;
      case ConstraintType::RelPose:
        for (auto& constraint : relative_pose_constraints_) {
          res_block_ids.emplace_back(constraint.getResidualBlockId());
        }
        break;
      default:
        break;
    }
    return res_block_ids;
  }

 private:
  Constraint::ConstraintId constraint_id_counter_ = 0;
  const Constraint::ConstraintId newConstraintId() {
    return constraint_id_counter_++;
  }

  std::list<AbsolutePoseConstraint> absolute_pose_constraints_;
  std::list<RelativePoseConstraint> relative_pose_constraints_;
  std::list<RegistrationConstraint> registration_constraints_;
  std::list<RelativePoseConstraint> submap_relative_pose_constraints_;
  std::list<RegistrationConstraint> force_registration_constraints_;
};
}  // namespace voxgraph

#endif  // VOXGRAPH_BACKEND_CONSTRAINT_CONSTRAINT_COLLECTION_H_
