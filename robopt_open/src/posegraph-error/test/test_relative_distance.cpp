#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <local-parameterization/quaternion-local-param.h>
#include <posegraph-error/relative-distance.h>

using robopt::posegraph::RelativeDistanceError;

namespace robopt {

class RelativeDistanceErrorTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RelativeDistanceErrorTerms() {
    srand((unsigned int) time(0));
  }

protected:
  typedef RelativeDistanceError ErrorTerm;

  virtual void SetUp() {
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    Eigen::Quaterniond q(Eigen::AngleAxisd(axis.norm(), axis.normalized()));
  }

  void GenerateRandomPose(
      double scale, Eigen::Quaterniond* q, Eigen::Vector3d* t) {
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    Eigen::Quaterniond q_tmp(Eigen::AngleAxisd(axis.norm(), axis.normalized()));
    (*q) = q_tmp;
    (*t) = Eigen::Vector3d::Random() * scale;
  }

  ceres::Problem problem_;
  ceres::Solver::Summary summary_;
  ceres::Solver::Options options_;

};


TEST_F(RelativeDistanceErrorTerms, Jacobians) {
  // Create the raw pointers to the data
  double** parameters_nom = new double*[4];
  double** parameters_dist = new double*[4];
  parameters_nom[0] = new double[defs::pose::kPoseBlockSize];
  parameters_nom[1] = new double[defs::pose::kPoseBlockSize];
  parameters_nom[2] = new double[defs::pose::kPositionBlockSize];
  parameters_nom[3] = new double[defs::pose::kPositionBlockSize];
  parameters_dist[0] = new double[defs::pose::kPoseBlockSize];
  parameters_dist[1] = new double[defs::pose::kPoseBlockSize];
  parameters_dist[2] = new double[defs::pose::kPositionBlockSize];
  parameters_dist[3] = new double[defs::pose::kPositionBlockSize];


  // Create Poses
  Eigen::Map<Eigen::Vector3d> p_W_S1(
        parameters_nom[0] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Vector3d> p_W_S2(
        parameters_nom[1] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q_W_S1(parameters_nom[0]);
  Eigen::Map<Eigen::Quaterniond> q_W_S2(parameters_nom[1]);
  Eigen::Map<Eigen::Vector3d> p_S_U1(parameters_nom[2]);
  Eigen::Map<Eigen::Vector3d> p_S_U2(parameters_nom[3]);

  Eigen::Quaterniond q_tmp;
  Eigen::Vector3d p_tmp;
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  q_W_S1 = q_tmp;
  p_W_S1 = p_tmp;
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  q_W_S2 = q_tmp;
  p_W_S2 = p_tmp;
  const Eigen::Matrix3d R_W_S1 = q_W_S1.toRotationMatrix();
  const Eigen::Matrix3d R_W_S2 = q_W_S2.toRotationMatrix();
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  p_S_U1 = p_tmp;
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  p_S_U2 = p_tmp;

  // Compute measurement (frame 2 into frame 1)
  const double measurement = 0.0;

  // Map the distorted parameters
  Eigen::Map<Eigen::Quaterniond> q_W_S1_dist(parameters_dist[0]);
  q_W_S1_dist = q_W_S1;
  Eigen::Map<Eigen::Vector3d> p_W_S1_dist(
        parameters_dist[0] + defs::pose::kOrientationBlockSize);
  p_W_S1_dist = p_W_S1;
  Eigen::Map<Eigen::Quaterniond> q_W_S2_dist(parameters_dist[1]);
  q_W_S2_dist = q_W_S2;
  Eigen::Map<Eigen::Vector3d> p_W_S2_dist(
        parameters_dist[1] + defs::pose::kOrientationBlockSize);
  p_W_S2_dist = p_W_S2;
  Eigen::Map<Eigen::Vector3d> p_S_U1_dist(parameters_dist[2]);
  p_S_U1_dist = p_S_U1;
  Eigen::Map<Eigen::Vector3d> p_S_U2_dist(parameters_dist[3]);
  p_S_U2_dist = p_S_U2;

  // Create the analytical jacobians
  double** jacobians_analytical = new double *[4];
  jacobians_analytical[0] = new double [defs::pose::kPoseBlockSize];
  jacobians_analytical[1] = new double [defs::pose::kPoseBlockSize];
  jacobians_analytical[2] = new double [defs::pose::kPositionBlockSize];
  jacobians_analytical[3] = new double [defs::pose::kPositionBlockSize];
  Eigen::Map<Eigen::Matrix<double,1, defs::pose::kPoseBlockSize,
      Eigen::RowMajor>> J_res_wrt_T_W_S1(jacobians_analytical[0]);
  Eigen::Map<Eigen::Matrix<double, 1, defs::visual::kPoseBlockSize,
      Eigen::RowMajor>> J_res_wrt_T_W_S2(jacobians_analytical[1]);
  Eigen::Map<Eigen::Matrix<double, 1, defs::pose::kPositionBlockSize,
      Eigen::RowMajor>> J_res_wrt_T_S_U1(jacobians_analytical[2]);
  Eigen::Map<Eigen::Matrix<double, 1, defs::pose::kPositionBlockSize,
      Eigen::RowMajor>> J_res_wrt_T_S_U2(jacobians_analytical[3]);

  // Compute the analytical derivatives
  const double sqrt_information = 0.24;
  ErrorTerm error_term(measurement, sqrt_information);
  double* residuals_nom = new double[1];
  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

  // Prepare the numerical jacobians
  double* residuals_dist = new double[defs::pose::kResidualSize];
  Eigen::Matrix<double, 1, defs::pose::kPoseBlockSize> J_res_wrt_T_W_S1_num;
  Eigen::Matrix<double, 1, defs::pose::kPoseBlockSize> J_res_wrt_T_W_S2_num;
  Eigen::Matrix<double, 1, defs::pose::kPositionBlockSize> J_res_wrt_T_S_U1_num;
  Eigen::Matrix<double, 1, defs::pose::kPositionBlockSize> J_res_wrt_T_S_U2_num;
  Eigen::Matrix<double, defs::pose::kResidualSize, 1>  difference;
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSize, 1>>
      residual_map_nom(residuals_nom);
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSize, 1>>
      residual_map_dist(residuals_dist);
  const double delta = 1e-7;

  // Jacobian for T_W_S1
  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Compute the disturbed parameters
    q_W_S1_dist = Eigen::Quaterniond(q_W_S1.w() + dist[3],
        q_W_S1.x() + dist[0], q_W_S1.y() + dist[1], q_W_S1.z() + dist[2]);
    q_W_S1_dist.normalize();
    p_W_S1_dist = p_W_S1 + dist.tail<3>();
    q_W_S2_dist = q_W_S2;
    p_W_S2_dist = p_W_S2;
    p_S_U1_dist = p_S_U1;
    p_S_U2_dist = p_S_U2;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);

    double difference = residual_map_dist(0) - residual_map_nom(0);
    J_res_wrt_T_W_S1_num(0, i) = difference / delta;
  }

  // Jacobian for T_W_S2
  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Compute the disturbed parameters
    q_W_S1_dist = q_W_S1;
    p_W_S1_dist = p_W_S1;
    q_W_S2_dist = Eigen::Quaterniond(q_W_S2.w() + dist[3],
        q_W_S2.x() + dist[0], q_W_S2.y() + dist[1], q_W_S2.z() + dist[2]);
    q_W_S2_dist.normalize();
    p_W_S2_dist = p_W_S2 + dist.tail<3>();
    p_S_U1_dist = p_S_U1;
    p_S_U2_dist = p_S_U2;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);

    double difference = residual_map_dist(0) - residual_map_nom(0);
    J_res_wrt_T_W_S2_num(0, i) = difference / delta;
  }

  // Jacobian for T_S_U1
  for (size_t i = 0; i < defs::pose::kPositionBlockSize; ++i) {
    Eigen::VectorXd dist(defs::pose::kPositionBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Compute the disturbed parameters
    q_W_S1_dist = q_W_S1;
    p_W_S1_dist = p_W_S1;
    q_W_S2_dist = q_W_S2;
    p_W_S2_dist = p_W_S2;
    p_S_U1_dist = p_S_U1 + dist;
    p_S_U2_dist = p_S_U2;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);

    double difference = residual_map_dist(0) - residual_map_nom(0);
    J_res_wrt_T_S_U1_num(0, i) = difference / delta;
  }

  // Jacobian for T_S_U2
  for (size_t i = 0; i < defs::pose::kPositionBlockSize; ++i) {
    Eigen::VectorXd dist(defs::pose::kPositionBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Compute the disturbed parameters
    q_W_S1_dist = q_W_S1;
    p_W_S1_dist = p_W_S1;
    q_W_S2_dist = q_W_S2;
    p_W_S2_dist = p_W_S2;
    p_S_U1_dist = p_S_U1;
    p_S_U2_dist = p_S_U2 + dist;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);

    double difference = residual_map_dist(0) - residual_map_nom(0);
    J_res_wrt_T_S_U2_num(0, i) = difference / delta;
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S1,
                          J_res_wrt_T_W_S1_num, 1e-5);
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S2,
                          J_res_wrt_T_W_S2_num, 1e-5);
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_S_U1,
                          J_res_wrt_T_S_U1_num, 1e-5);
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_S_U2,
                          J_res_wrt_T_S_U2_num, 1e-5);
}

} // namespace robopt


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
