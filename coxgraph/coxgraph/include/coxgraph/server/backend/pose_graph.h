#ifndef COXGRAPH_SERVER_BACKEND_POSE_GRAPH_H_
#define COXGRAPH_SERVER_BACKEND_POSE_GRAPH_H_

#include <list>
#include <map>
#include <memory>

#include "coxgraph/common.h"
#include "coxgraph/server/backend/client_frame_node.h"
#include "coxgraph/server/backend/constraint_collection.h"
#include "coxgraph/server/backend/node_collection.h"

namespace coxgraph {
namespace server {
class PoseGraph {
 public:
  typedef std::shared_ptr<const PoseGraph> ConstPtr;
  typedef std::list<ceres::Solver::Summary> SolverSummaryList;
  typedef std::map<const CliId, const Transformation> PoseMap;

  PoseGraph() = default;
  ~PoseGraph() = default;

  void addClientNode(const ClientFrameNode::Config& config) {
    node_collection_.addClientNode(config);
  }
  bool hasClientNode(const CliId& cid) {
    auto ptr = node_collection_.getClientNodePtrById(cid);
    return ptr != nullptr;
  }

  void addClientRelativePoseConstraint(
      const RelativePoseConstraint::Config& config) {
    constraint_collection_.addClientRelativePoseConstraint(config);
  }

  void resetClientRelativePoseConstraint() {
    constraint_collection_.resetClientRelativePoseConstraint();
  }

  void initialize() {
    // Initialize the problem
    problem_options_.local_parameterization_ownership =
        ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problem_ptr_.reset(new ceres::Problem(problem_options_));

    // Add the appropriate constraints
    constraint_collection_.addConstraintsToProblem(node_collection_,
                                                   problem_ptr_.get());
  }

  void optimize() {
    // Initialize the problem
    initialize();

    // Run the solver
    ceres::Solver::Options ceres_options;
    // TODO(victorr): Set these from parameters
    // TODO(victorr): Look into manual parameter block ordering
    ceres_options.parameter_tolerance = 3e-3;
    //  ceres_options.max_num_iterations = 4;
    ceres_options.max_solver_time_in_seconds = 4;
    ceres_options.num_threads = 4;
    ceres_options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    // NOTE: For small problems DENSE_SCHUR is much faster

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, problem_ptr_.get(), &summary);

    // Display and store the solver summary
    std::cout << summary.BriefReport() << std::endl;
    solver_summaries_.emplace_back(summary);
  }

  PoseMap getClientMapTf() {
    PoseMap client_map_tf;
    for (const auto& client_node_kv : node_collection_.getClientNodes()) {
      client_map_tf.emplace(client_node_kv.second->getCliId(),
                            client_node_kv.second->getPose());
    }
    return client_map_tf;
  }

 private:
  NodeCollection node_collection_;
  ConstraintCollection constraint_collection_;

  // Ceres problem
  ceres::Problem::Options problem_options_;
  std::shared_ptr<ceres::Problem> problem_ptr_;
  SolverSummaryList solver_summaries_;
};

}  // namespace server
}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_BACKEND_POSE_GRAPH_H_
