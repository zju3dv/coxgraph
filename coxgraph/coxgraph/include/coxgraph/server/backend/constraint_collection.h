#ifndef COXGRAPH_SERVER_BACKEND_CONSTRAINT_COLLECTION_H_
#define COXGRAPH_SERVER_BACKEND_CONSTRAINT_COLLECTION_H_

#include <voxgraph/backend/constraint/constraint.h>

#include <list>

#include "coxgraph/server/backend/node_collection.h"
#include "coxgraph/server/backend/relative_pose_constraint.h"

namespace coxgraph {

namespace server {

class ConstraintCollection {
 public:
  ConstraintCollection() = default;
  ~ConstraintCollection() = default;

  void addClientRelativePoseConstraint(
      const RelativePoseConstraint::Config& config) {
    client_relative_pose_constraints_.emplace_back(newConstraintId(), config);
  }

  void resetClientRelativePoseConstraint() {
    client_relative_pose_constraints_.clear();
  }

  void addConstraintsToProblem(const NodeCollection& node_collection,
                               ceres::Problem* problem_ptr) {
    for (RelativePoseConstraint& client_relative_pose_constraint :
         client_relative_pose_constraints_) {
      client_relative_pose_constraint.addToProblem(node_collection,
                                                   problem_ptr);
    }
  }

 private:
  using Constraint = voxgraph::Constraint;

  Constraint::ConstraintId constraint_id_counter_ = 0;
  const Constraint::ConstraintId newConstraintId() {
    return constraint_id_counter_++;
  }

  std::list<RelativePoseConstraint> client_relative_pose_constraints_;
};

}  // namespace server

}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_BACKEND_CONSTRAINT_COLLECTION_H_
