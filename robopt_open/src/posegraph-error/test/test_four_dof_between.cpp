#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <common/common.h>
#include <local-parameterization/pose-quaternion-yaw-local-param.h>
#include <posegraph-error/four-dof-between.h>

using robopt::posegraph::FourDofBetweenError;

namespace robopt {

class FourDofBetweenErrorTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FourDofBetweenErrorTerms() {
    srand((unsigned int) time(0));
  }

protected:
  typedef FourDofBetweenError ErrorTerm;

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


TEST_F(FourDofBetweenErrorTerms, JacobiansImu) {
  // Create the raw pointers to the data
  double** parameters_nom = new double*[4];
  double** parameters_dist = new double*[4];
  for (size_t i = 0; i < 4; ++i) {
    parameters_nom[i] = new double [defs::pose::kPoseBlockSize];
    parameters_dist[i] = new double [defs::pose::kPoseBlockSize];
  }


  // Create Poses
  Eigen::Map<Eigen::Vector3d> p_W_S1(
        parameters_nom[0] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Vector3d> p_W_S2(
        parameters_nom[1] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q_W_S1(parameters_nom[0]);
  Eigen::Map<Eigen::Quaterniond> q_W_S2(parameters_nom[1]);
  Eigen::Map<Eigen::Vector3d> p_S_C1(
        parameters_nom[2] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Vector3d> p_S_C2(
        parameters_nom[3] + defs::pose::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q_S_C1(parameters_nom[2]);
  Eigen::Map<Eigen::Quaterniond> q_S_C2(parameters_nom[3]);
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
  q_S_C1 = q_tmp;
  p_S_C1 = p_tmp;
  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
  q_S_C2 = q_tmp;
  p_S_C2 = p_tmp;
  const Eigen::Matrix3d R_S_C1 = q_S_C1.toRotationMatrix();
  const Eigen::Matrix3d R_S_C2 = q_S_C2.toRotationMatrix();

  // Compute measurement (frame 2 into frame 1)
  Eigen::Vector3d translation_measurement = p_W_S1 - p_W_S2 +
      Eigen::Vector3d::Random() * 0.2;;
  const double yaw_S1 = common::yaw::LogMap(q_W_S1);
  const double yaw_S2 = common::yaw::LogMap(q_W_S2);
  double rotation_measurement = common::yaw::normalizeYaw(yaw_S1 - yaw_S2) +
      0.2;

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
  Eigen::Map<Eigen::Quaterniond> q_S_C1_dist(parameters_dist[2]);
  q_S_C1_dist = q_S_C1;
  Eigen::Map<Eigen::Vector3d> p_S_C1_dist(
        parameters_dist[2] + defs::pose::kOrientationBlockSize);
  p_S_C1_dist = p_S_C1;
  Eigen::Map<Eigen::Quaterniond> q_S_C2_dist(parameters_dist[3]);
  q_S_C2_dist = q_S_C2;
  Eigen::Map<Eigen::Vector3d> p_S_C2_dist(
        parameters_dist[3] + defs::pose::kOrientationBlockSize);
  p_S_C2_dist = p_S_C2;

  // Create the analytical jacobians
  double** jacobians_analytical = new double *[4];
  for (size_t i = 0; i < 4; ++i) {
    jacobians_analytical[i] = new double [defs::pose::kResidualSizeYaw *
        defs::pose::kPoseBlockSize];
  }
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
            defs::pose::kPoseBlockSize, Eigen::RowMajor>>
      J_res_wrt_T_W_S1(jacobians_analytical[0]);
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
      J_res_wrt_T_W_S2(jacobians_analytical[1]);
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
            defs::pose::kPoseBlockSize, Eigen::RowMajor>>
      J_res_wrt_T_S_C1(jacobians_analytical[2]);
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
            defs::pose::kPoseBlockSize, Eigen::RowMajor>>
      J_res_wrt_T_S_C2(jacobians_analytical[3]);

  // Compute the analytical derivatives
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
      defs::pose::kResidualSizeYaw> sqrt_information;
  sqrt_information.setIdentity();
  sqrt_information(0,0) = 3.0;
  sqrt_information(1,1) = sqrt_information(2,2) = sqrt_information(3,3) =
      0.5;
  ErrorTerm error_term(rotation_measurement,
                       translation_measurement,
                       sqrt_information,
                       defs::pose::PoseErrorType::kImu);
  double* residuals_nom = new double[defs::pose::kResidualSize];
  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

  // Prepare the numerical jacobians
  double* residuals_dist = new double[defs::pose::kResidualSizeYaw];
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
               defs::pose::kPoseBlockSize>
      J_res_wrt_T_W_S1_num;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
               defs::pose::kPoseBlockSize>
      J_res_wrt_T_W_S2_num;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
               defs::pose::kPoseBlockSize>
      J_res_wrt_T_S_C1_num;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
               defs::pose::kPoseBlockSize>
      J_res_wrt_T_S_C2_num;
  Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>  difference;
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>>
      residual_map_nom(residuals_nom);
  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>>
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
    //q_W_S1_dist.normalize();
    p_W_S1_dist = p_W_S1 + dist.tail<3>();
    q_W_S2_dist = q_W_S2;
    p_W_S2_dist = p_W_S2;
    q_S_C1_dist = q_S_C1;
    p_S_C1_dist = p_S_C1;
    q_S_C2_dist = q_S_C2;
    p_S_C2_dist = p_S_C2;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);

    difference(0) = residual_map_dist(0) - residual_map_nom(0);
    difference.tail<3>() = residual_map_dist.tail<3>() -
        residual_map_nom.tail<3>();

    J_res_wrt_T_W_S1_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
        difference / delta;
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
    // q_W_S2_dist.normalize(); // Why does this lead to wrong results???
    p_W_S2_dist = p_W_S2 + dist.tail<3>();
    q_S_C1_dist = q_S_C1;
    p_S_C1_dist = p_S_C1;
    q_S_C2_dist = q_S_C2;
    p_S_C2_dist = p_S_C2;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference(0) = residual_map_dist(0) - residual_map_nom(0);
    difference.tail<3>() = residual_map_dist.tail<3>() -
        residual_map_nom.tail<3>();

    J_res_wrt_T_W_S2_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
        difference / delta;
  }

  // Jacobian for T_S_C1
  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Compute the disturbed parameters
    q_W_S1_dist = q_W_S1;
    p_W_S1_dist = p_W_S1;
    q_W_S2_dist = q_W_S2;
    p_W_S2_dist = p_W_S2;
    q_S_C1_dist = Eigen::Quaterniond(q_S_C1.w() + dist[3],
        q_S_C1.x() + dist[0], q_S_C1.y() + dist[1], q_S_C1.z() + dist[2]);
    // q_S_C1_dist.normalize(); // Why does this lead to wrong results???
    p_S_C1_dist = p_S_C1 + dist.tail<3>();
    q_S_C2_dist = q_S_C2;
    p_S_C2_dist = p_S_C2;

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference(0) = residual_map_dist(0) - residual_map_nom(0);
    difference.tail<3>() = residual_map_dist.tail<3>() -
        residual_map_nom.tail<3>();

    J_res_wrt_T_S_C1_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
        difference / delta;
  }


  // Jacobian for T_S_C2
  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Compute the disturbed parameters
    q_W_S1_dist = q_W_S1;
    p_W_S1_dist = p_W_S1;
    q_W_S2_dist = q_W_S2;
    p_W_S2_dist = p_W_S2;
    q_S_C1_dist = q_S_C1;
    p_S_C1_dist = p_S_C1;
    q_S_C2_dist = Eigen::Quaterniond(q_S_C2.w() + dist[3],
        q_S_C2.x() + dist[0], q_S_C2.y() + dist[1], q_S_C2.z() + dist[2]);
    // q_S_C2_dist.normalize(); // Why does this lead to wrong results???
    p_S_C2_dist = p_S_C2 + dist.tail<3>();

    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference(0) = residual_map_dist(0) - residual_map_nom(0);
    difference.tail<3>() = residual_map_dist.tail<3>() -
        residual_map_nom.tail<3>();

    J_res_wrt_T_S_C2_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
        difference / delta;
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S1,
                          J_res_wrt_T_W_S1_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S2,
                          J_res_wrt_T_W_S2_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_S_C1,
                          J_res_wrt_T_S_C1_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_S_C2,
                          J_res_wrt_T_S_C2_num, 1e-4);
}

TEST_F(FourDofBetweenErrorTerms, JacobiansCam) {
  // Create the raw pointers to the data
//  double** parameters_nom = new double*[4];
//  double** parameters_dist = new double*[4];
//  for (size_t i = 0; i < 4; ++i) {
//    parameters_nom[i] = new double [defs::pose::kPoseBlockSize];
//    parameters_dist[i] = new double [defs::pose::kPoseBlockSize];
//  }


//  // Create Poses
//  Eigen::Map<Eigen::Vector3d> p_W_S1(
//        parameters_nom[0] + defs::pose::kOrientationBlockSize);
//  Eigen::Map<Eigen::Vector3d> p_W_S2(
//        parameters_nom[1] + defs::pose::kOrientationBlockSize);
//  Eigen::Map<Eigen::Quaterniond> q_W_S1(parameters_nom[0]);
//  Eigen::Map<Eigen::Quaterniond> q_W_S2(parameters_nom[1]);
//  Eigen::Map<Eigen::Vector3d> p_S_C1(
//        parameters_nom[2] + defs::pose::kOrientationBlockSize);
//  Eigen::Map<Eigen::Vector3d> p_S_C2(
//        parameters_nom[3] + defs::pose::kOrientationBlockSize);
//  Eigen::Map<Eigen::Quaterniond> q_S_C1(parameters_nom[2]);
//  Eigen::Map<Eigen::Quaterniond> q_S_C2(parameters_nom[3]);
//  Eigen::Quaterniond q_tmp;
//  Eigen::Vector3d p_tmp;
//  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
//  q_W_S1 = q_tmp;
//  p_W_S1 = p_tmp;
//  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
//  q_W_S2 = q_tmp;
//  p_W_S2 = p_tmp;
//  const Eigen::Matrix3d R_W_S1 = q_W_S1.toRotationMatrix();
//  const Eigen::Matrix3d R_W_S2 = q_W_S2.toRotationMatrix();
//  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
//  q_S_C1 = q_tmp;
//  p_S_C1 = p_tmp;
//  GenerateRandomPose(0.5, &q_tmp, &p_tmp);
//  q_S_C2 = q_tmp;
//  p_S_C2 = p_tmp;
//  const Eigen::Matrix3d R_S_C1 = q_S_C1.toRotationMatrix();
//  const Eigen::Matrix3d R_S_C2 = q_S_C2.toRotationMatrix();

//  // Compute measurement (frame 2 into frame 1)
//  Eigen::Quaterniond q_W_C1 = q_W_S1 * q_S_C1;
//  Eigen::Vector3d p_W_C1 = q_W_S1 * p_S_C1 + p_W_S1;
//  Eigen::Quaterniond q_W_C2 = q_W_S2 * q_S_C2;
//  Eigen::Vector3d p_W_C2 = q_W_S2 * p_S_C2 + p_W_S2;

//  Eigen::Vector3d translation_measurement = p_W_C1 - p_W_C2 +
//      Eigen::Vector3d::Random() * 0.2;
//  const double yaw_S1 = common::yaw::LogMap(q_W_S1);
//  const double yaw_S2 = common::yaw::LogMap(q_W_S2);
//  double rotation_measurement = common::yaw::normalizeYaw(yaw_S1 - yaw_S2) +
//      0.2;

//  // Map the distorted parameters
//  Eigen::Map<Eigen::Quaterniond> q_W_S1_dist(parameters_dist[0]);
//  q_W_S1_dist = q_W_S1;
//  Eigen::Map<Eigen::Vector3d> p_W_S1_dist(
//        parameters_dist[0] + defs::pose::kOrientationBlockSize);
//  p_W_S1_dist = p_W_S1;
//  Eigen::Map<Eigen::Quaterniond> q_W_S2_dist(parameters_dist[1]);
//  q_W_S2_dist = q_W_S2;
//  Eigen::Map<Eigen::Vector3d> p_W_S2_dist(
//        parameters_dist[1] + defs::pose::kOrientationBlockSize);
//  p_W_S2_dist = p_W_S2;
//  Eigen::Map<Eigen::Quaterniond> q_S_C1_dist(parameters_dist[2]);
//  q_S_C1_dist = q_S_C1;
//  Eigen::Map<Eigen::Vector3d> p_S_C1_dist(
//        parameters_dist[2] + defs::pose::kOrientationBlockSize);
//  p_S_C1_dist = p_S_C1;
//  Eigen::Map<Eigen::Quaterniond> q_S_C2_dist(parameters_dist[3]);
//  q_S_C2_dist = q_S_C2;
//  Eigen::Map<Eigen::Vector3d> p_S_C2_dist(
//        parameters_dist[3] + defs::pose::kOrientationBlockSize);
//  p_S_C2_dist = p_S_C2;

//  // Create the analytical jacobians
//  double** jacobians_analytical = new double *[4];
//  for (size_t i = 0; i < 4; ++i) {
//    jacobians_analytical[i] = new double [defs::pose::kResidualSizeYaw *
//        defs::pose::kPoseBlockSize];
//  }
//  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//            defs::pose::kPoseBlockSize, Eigen::RowMajor>>
//      J_res_wrt_T_W_S1(jacobians_analytical[0]);
//  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
//      J_res_wrt_T_W_S2(jacobians_analytical[1]);
//  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//            defs::pose::kPoseBlockSize, Eigen::RowMajor>>
//      J_res_wrt_T_S_C1(jacobians_analytical[2]);
//  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//            defs::pose::kPoseBlockSize, Eigen::RowMajor>>
//      J_res_wrt_T_S_C2(jacobians_analytical[3]);

//  // Compute the analytical derivatives
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//      defs::pose::kResidualSizeYaw> sqrt_information;
//  sqrt_information.setIdentity();
//  sqrt_information(0,0) = 3.0;
//  sqrt_information(1,1) = sqrt_information(2,2) = sqrt_information(3,3) =
//      0.5;
//  ErrorTerm error_term(rotation_measurement,
//                       translation_measurement,
//                       sqrt_information,
//                       defs::pose::PoseErrorType::kVisual);
//  double* residuals_nom = new double[defs::pose::kResidualSize];
//  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

//  // Prepare the numerical jacobians
//  double* residuals_dist = new double[defs::pose::kResidualSizeYaw];
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//               defs::pose::kPoseBlockSize>
//      J_res_wrt_T_W_S1_num;
//  J_res_wrt_T_W_S1_num.setZero();
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//               defs::pose::kPoseBlockSize>
//      J_res_wrt_T_W_S2_num;
//  J_res_wrt_T_W_S2_num.setZero();
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//               defs::pose::kPoseBlockSize>
//      J_res_wrt_T_S_C1_num;
//  J_res_wrt_T_S_C1_num.setZero();
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw,
//               defs::pose::kPoseBlockSize>
//      J_res_wrt_T_S_C2_num;
//  J_res_wrt_T_S_C2_num.setZero();
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>  difference;
//  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>>
//      residual_map_nom(residuals_nom);
//  Eigen::Map<Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 1>>
//      residual_map_dist(residuals_dist);
//  const double delta = 1e-7;

//  // Jacobian for T_W_S1
//  Eigen::Matrix<double, defs::pose::kResidualSizeYaw, 4>
//      J_res_wrt_T_W_S1_min_num;
//  J_res_wrt_T_W_S1_min_num.setZero();
//  for (size_t i = 0; i < 4/*defs::pose::kPoseBlockSize*/; ++i) {
//    Eigen::VectorXd dist(4/*defs::pose::kPoseBlockSize*/);
//    dist.setZero();
//    dist[i] = delta;

//    // Compute the disturbed parameters
//    double delta_yaw = dist(0);
//    Eigen::Vector4d q_W_S1_vec = q_W_S1.coeffs();
//    Eigen::Quaterniond q_W_S1_dist2;
//    common::yaw::Plus(q_W_S1_vec, delta_yaw, &q_W_S1_dist2);
//    q_W_S1_dist = q_W_S1_dist2;
//        /*Eigen::Quaterniond(q_W_S1.w() + dist[3],
//        q_W_S1.x() + dist[0], q_W_S1.y() + dist[1], q_W_S1.z() + dist[2]);
//    q_W_S1_dist.normalize();
//    q_W_S1_dist.coeffs() = q_W_S1_dist.coeffs()/(q_W_S1_dist.norm());
//    Eigen::Vector4d diff_q = q_W_S1_dist.coeffs() - q_W_S1.coeffs();*/

//    p_W_S1_dist = p_W_S1 + dist.tail<3>();
//    q_W_S2_dist = q_W_S2;
//    p_W_S2_dist = p_W_S2;
//    q_S_C1_dist = q_S_C1;
//    p_S_C1_dist = p_S_C1;
//    q_S_C2_dist = q_S_C2;
//    p_S_C2_dist = p_S_C2;

//    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
//    difference(0) = residual_map_dist(0) - residual_map_nom(0);
//    difference.tail<3>() = residual_map_dist.tail<3>() -
//        residual_map_nom.tail<3>();
//    /*J_res_wrt_T_W_S1_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
//        difference / delta;*/
//    J_res_wrt_T_W_S1_min_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
//        difference / delta;
//  }
//  local_param::PoseQuaternionYawLocalParameterization local_parameterization;
//  Eigen::Matrix<double, 7, 4, Eigen::RowMajor> J_tmp;
//  local_parameterization.ComputeJacobian(q_W_S1.coeffs().data(), J_tmp.data());
//  Eigen::Matrix<double, 4, 7> J_tmp_inv =
//      (J_tmp.transpose() * J_tmp).inverse() * J_tmp.transpose();
//  std::cout << "J_tmp_inv: " << J_tmp_inv << std::endl;
//  J_res_wrt_T_W_S1_num = J_res_wrt_T_W_S1_min_num * J_tmp_inv;
//  std::cout << "J_res_wrt_T_W_S1_min:\n" << J_res_wrt_T_W_S1_min_num << std::endl;

//  // Jacobian for T_W_S2
//  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
//    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
//    dist.setZero();
//    dist[i] = delta;

//    // Compute the disturbed parameters
//    q_W_S1_dist = q_W_S1;
//    p_W_S1_dist = p_W_S1;
//    q_W_S2_dist = Eigen::Quaterniond(q_W_S2.w() + dist[3],
//        q_W_S2.x() + dist[0], q_W_S2.y() + dist[1], q_W_S2.z() + dist[2]);
//    // q_W_S2_dist.normalize(); // Why does this lead to wrong results???
//    p_W_S2_dist = p_W_S2 + dist.tail<3>();
//    q_S_C1_dist = q_S_C1;
//    p_S_C1_dist = p_S_C1;
//    q_S_C2_dist = q_S_C2;
//    p_S_C2_dist = p_S_C2;

//    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
//    difference(0) = residual_map_dist(0) - residual_map_nom(0);
//    difference.tail<3>() = residual_map_dist.tail<3>() -
//        residual_map_nom.tail<3>();

//    J_res_wrt_T_W_S2_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
//        difference / delta;
//  }

//  // Jacobian for T_S_C1
//  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
//    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
//    dist.setZero();
//    dist[i] = delta;

//    // Compute the disturbed parameters
//    q_W_S1_dist = q_W_S1;
//    p_W_S1_dist = p_W_S1;
//    q_W_S2_dist = q_W_S2;
//    p_W_S2_dist = p_W_S2;
//    q_S_C1_dist = Eigen::Quaterniond(q_S_C1.w() + dist[3],
//        q_S_C1.x() + dist[0], q_S_C1.y() + dist[1], q_S_C1.z() + dist[2]);
//    // q_S_C1_dist.normalize(); // Why does this lead to wrong results???
//    p_S_C1_dist = p_S_C1 + dist.tail<3>();
//    q_S_C2_dist = q_S_C2;
//    p_S_C2_dist = p_S_C2;

//    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
//    difference(0) = residual_map_dist(0) - residual_map_nom(0);
//    difference.tail<3>() = residual_map_dist.tail<3>() -
//        residual_map_nom.tail<3>();

//    J_res_wrt_T_S_C1_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
//        difference / delta;
//  }


//  // Jacobian for T_S_C2
//  for (size_t i = 0; i < defs::pose::kPoseBlockSize; ++i) {
//    Eigen::VectorXd dist(defs::pose::kPoseBlockSize);
//    dist.setZero();
//    dist[i] = delta;

//    // Compute the disturbed parameters
//    q_W_S1_dist = q_W_S1;
//    p_W_S1_dist = p_W_S1;
//    q_W_S2_dist = q_W_S2;
//    p_W_S2_dist = p_W_S2;
//    q_S_C1_dist = q_S_C1;
//    p_S_C1_dist = p_S_C1;
//    q_S_C2_dist = Eigen::Quaterniond(q_S_C2.w() + dist[3],
//        q_S_C2.x() + dist[0], q_S_C2.y() + dist[1], q_S_C2.z() + dist[2]);
//    // q_S_C2_dist.normalize(); // Why does this lead to wrong results???
//    p_S_C2_dist = p_S_C2 + dist.tail<3>();

//    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
//    difference(0) = residual_map_dist(0) - residual_map_nom(0);
//    difference.tail<3>() = residual_map_dist.tail<3>() -
//        residual_map_nom.tail<3>();

//    J_res_wrt_T_S_C2_num.block<defs::pose::kResidualSizeYaw, 1>(0, i) =
//        difference / delta;
//  }

//  std::cout << "J_res_wrt_T_W_S1_num:\n" << J_res_wrt_T_W_S1_num << std::endl;
//  std::cout << "J_res_wrt_T_W_S1:\n" << J_res_wrt_T_W_S1 << std::endl;
//  std::cout << "J_res_wrt_T_W_S2_num:\n" << J_res_wrt_T_W_S2_num << std::endl;
//  std::cout << "J_res_wrt_T_S_C1_num:\n" << J_res_wrt_T_S_C1_num << std::endl;
//  std::cout << "J_res_wrt_T_S_C2_num:\n" << J_res_wrt_T_S_C2_num << std::endl;

//  // Carry out the checks
//  /*CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S1,
//                          J_res_wrt_T_W_S1_num, 1e-4);
//  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S2,
//                          J_res_wrt_T_W_S2_num, 1e-4);
//  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_S_C1,
//                          J_res_wrt_T_S_C1_num, 1e-4);
//  CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_S_C2,
//                          J_res_wrt_T_S_C2_num, 1e-4);*/
}


} // namespace robopt


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
