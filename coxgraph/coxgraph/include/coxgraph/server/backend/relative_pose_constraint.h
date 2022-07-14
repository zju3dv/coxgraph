#ifndef COXGRAPH_SERVER_BACKEND_RELATIVE_POSE_CONSTRAINT_H_
#define COXGRAPH_SERVER_BACKEND_RELATIVE_POSE_CONSTRAINT_H_

#include <ceres/ceres.h>
#include <voxgraph/backend/constraint/cost_functions/relative_pose_cost_function.h>

#include <memory>

#include "coxgraph/common.h"
#include "coxgraph/server/backend/node_collection.h"

namespace coxgraph {
namespace server {
class Constraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<Constraint> Ptr;
  typedef unsigned int ConstraintId;
  typedef Eigen::Matrix<double, 4, 4> InformationMatrix;

  struct Config {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InformationMatrix information_matrix;
    bool allow_semi_definite_information_matrix = false;
  };

  explicit Constraint(ConstraintId constraint_id, const Config& config)
      : constraint_id_(constraint_id) {
    if (!config.allow_semi_definite_information_matrix) {
      // Compute the square root of the information matrix
      // using Eigen's Cholesky LL^T decomposition
      Eigen::LLT<Eigen::MatrixXd> information_llt(config.information_matrix);
      CHECK(information_llt.info() != Eigen::NumericalIssue)
          << "The square root of the information matrix could not be computed, "
          << "make sure it is symmetric and positive definite: "
          << config.information_matrix;
      sqrt_information_matrix_ = information_llt.matrixL();
    } else {
      // Compute the robust square root of the information matrix
      // using Eigen's LDL^T Cholesky decomposition
      Eigen::LDLT<Eigen::MatrixXd> information_ldlt(config.information_matrix);
      CHECK(information_ldlt.info() != Eigen::NumericalIssue)
          << "The square root of the information matrix could not be computed,"
          << "despite using the robust LDL^T Cholesky decomposition, check: "
          << config.information_matrix;
      CHECK(information_ldlt.isPositive())
          << "The information matrix must be positive semi-definite:"
          << config.information_matrix;

      sqrt_information_matrix_ =
          information_ldlt.transpositionsP().transpose() *
          information_ldlt.matrixL().toDenseMatrix() *
          information_ldlt.vectorD().cwiseSqrt().asDiagonal() *
          information_ldlt.transpositionsP();
      // NOTE: The first permutation term (transpositionsP.transpose) could
      //       be left out, since it cancels once the residual is squared.
      //       However, we leave it in such that the sqrt_information_matrix_
      //       looks familiar to users debugging intermediate steps.
    }
  }
  virtual ~Constraint() = default;

  virtual void addToProblem(const NodeCollection& node_collection,
                            ceres::Problem* problem) = 0;

  const ceres::ResidualBlockId getResidualBlockId() {
    return residual_block_id_;
  }

 protected:
  static constexpr ceres::LossFunction* kNoRobustLossFunction = nullptr;

  const ConstraintId constraint_id_;
  ceres::ResidualBlockId residual_block_id_ = nullptr;

  InformationMatrix sqrt_information_matrix_;
};

class RelativePoseConstraint : public Constraint {
 public:
  struct Config : Constraint::Config {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CliId origin_client_id;
    CliId destination_client_id;
    Transformation T_origin_destination;
  };

  RelativePoseConstraint(ConstraintId constraint_id, const Config& config)
      : Constraint(constraint_id, config), config_(config) {}
  ~RelativePoseConstraint() = default;

  void addToProblem(const NodeCollection& node_collection,
                    ceres::Problem* problem) {
    CHECK_NOTNULL(problem);

    ceres::LossFunction* loss_function = kNoRobustLossFunction;

    // Get pointers to both submap nodes
    ClientFrameNode::Ptr origin_client_node_ptr =
        node_collection.getClientNodePtrById(config_.origin_client_id);
    ClientFrameNode::Ptr destination_client_node_ptr =
        node_collection.getClientNodePtrById(config_.destination_client_id);
    CHECK_NOTNULL(origin_client_node_ptr);
    CHECK_NOTNULL(destination_client_node_ptr);

    // Add the submap parameters to the problem
    origin_client_node_ptr->addToProblem(
        problem, node_collection.getLocalParameterization());
    destination_client_node_ptr->addToProblem(
        problem, node_collection.getLocalParameterization());

    // Add the constraint to the optimization and keep track of it
    ceres::CostFunction* cost_function = RelativePoseCostFunction::Create(
        config_.T_origin_destination, sqrt_information_matrix_);
    residual_block_id_ = problem->AddResidualBlock(
        cost_function, loss_function,
        origin_client_node_ptr->getPosePtr()->optimizationVectorData(),
        destination_client_node_ptr->getPosePtr()->optimizationVectorData());
  }

 private:
  using RelativePoseCostFunction = voxgraph::RelativePoseCostFunction;

  const Config config_;
};

}  // namespace server
}  // namespace coxgraph
#endif  // COXGRAPH_SERVER_BACKEND_RELATIVE_POSE_CONSTRAINT_H_
