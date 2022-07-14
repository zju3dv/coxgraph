#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <common/definitions.h>
#include <local-parameterization/pose-quaternion-yaw-local-param.h>
#include <common/common.h>

using robopt::local_param::PoseQuaternionYawLocalParameterization;

namespace robopt {

class PoseQuaternionYawLocalParameterizationTest: public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PoseQuaternionYawLocalParameterizationTest() {
    srand((unsigned int) time(0));
  }

protected:
  typedef PoseQuaternionYawLocalParameterization LocalParam;

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
};

TEST_F(PoseQuaternionYawLocalParameterizationTest, Plus) {
  double** parameters = new double*[1];
  parameters[0] = new double [defs::pose::kPoseBlockSize];
  double** parameters_new = new double*[1];
  parameters_new[0] = new double [defs::pose::kPoseBlockSize];

  // Create Poses
  Eigen::Map<Eigen::Vector3d> p(
        parameters[0] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q(parameters[0]);
  Eigen::Map<Eigen::Vector3d> p_new(
          parameters_new[0] + defs::pose::kOrientationBlockSize);
    Eigen::Map<Eigen::Quaterniond> q_new(parameters_new[0]);

  Eigen::Quaterniond q_tmp;
  Eigen::Vector3d p_tmp;
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  q = q_tmp;
  p = p_tmp;

  const double delta_yaw = 0.3;
  double* delta = new double[4];
  delta[0] = delta_yaw;
  delta[1] = 0.0;
  delta[2] = 0.0;
  delta[3] = 0.0;

  // Perform the local plus operation
  LocalParam local_parameterization;
  local_parameterization.Plus(parameters[0], delta, parameters_new[0]);

  // Compute the old and new angles
  Eigen::Matrix< double, 3, 1> rpy = q.toRotationMatrix().eulerAngles(2,1,0);
  const double yaw = ((double)rpy(0));
  const double pitch = ((double)rpy(1));
  const double roll = ((double)rpy(2));
  Eigen::Matrix< double, 3, 1> rpy_new =
      q_new.toRotationMatrix().eulerAngles(2,1,0);
  const double yaw_new = ((double)rpy_new(0));
  const double pitch_new = ((double)rpy_new(1));
  const double roll_new = ((double)rpy_new(2));

  CHECK_EIGEN_MATRIX_NEAR(p, p_new, 1e-5);
  CHECK_NEAR(roll, roll_new, 1e-5);
  CHECK_NEAR(pitch, pitch_new, 1e-5);
  CHECK_NEAR(yaw + delta_yaw, yaw_new, 1e-5);
}


TEST_F(PoseQuaternionYawLocalParameterizationTest, Jacobian) {
  // Create the raw pointers to the data
  double** parameters_nom = new double*[1];
  parameters_nom[0] = new double [defs::pose::kPoseBlockSize];
  double** parameters_dist = new double*[1];
  parameters_dist[0] = new double [defs::pose::kPoseBlockSize];

  // Create Poses
  Eigen::Map<Eigen::Vector3d> p(
        parameters_nom[0] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q(parameters_nom[0]);

  Eigen::Quaterniond q_tmp;
  Eigen::Vector3d p_tmp;
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  q = q_tmp;
  p = p_tmp;


  // Map the distorted parameters
  Eigen::Map<Eigen::Quaterniond> q_dist(parameters_dist[0]);
  q_dist = q;
  Eigen::Map<Eigen::Vector3d> p_dist(
        parameters_dist[0] + defs::pose::kOrientationBlockSize);
  p_dist = p;

  // Create the analytical jacobians
  double** jacobians_analytical = new double *[1];
  jacobians_analytical[0] = new double [defs::pose::kPoseBlockSize * 4];

  Eigen::Map<Eigen::Matrix<double, defs::pose::kPoseBlockSize,4,
      Eigen::RowMajor>> J_map_wrt_T(jacobians_analytical[0]);

  // Compute the analytical derivatives
  LocalParam local_parameterization;
  local_parameterization.ComputeJacobian(parameters_nom[0],
      jacobians_analytical[0]);

  // Prepare the numerical jacobians
  double* delta_x = new double[4];
  double* delta_T = new double[defs::pose::kPoseBlockSize];
  Eigen::Matrix<double, defs::pose::kPoseBlockSize, 4>
      J_map_wrt_T_num;
  const double delta = 1e-5;
  Eigen::Map<Eigen::Matrix<double, 4, 1>>
      delta_x_map(delta_x);

  // Compute the numerical jacobian
  for (size_t i = 0; i < 4; ++i) {
    delta_x_map.setZero();
    delta_x_map(i,0) = delta;
    local_parameterization.Plus(parameters_nom[0], delta_x, parameters_dist[0]);
    for (size_t k = 0; k < defs::pose::kPoseBlockSize; ++k) {
      J_map_wrt_T_num(k, i) = (parameters_dist[0][k] - parameters_nom[0][k]) /
          delta;
    }
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_map_wrt_T,
                          J_map_wrt_T_num, 1e-4);
}


} // namespace robopt


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
