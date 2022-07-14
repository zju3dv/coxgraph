#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/distortion-equidistant.h>

#include <local-parameterization/quaternion-local-param.h>
#include <reprojection-error/local-euclidean.h>

using robopt::reprojection::LocalEuclideanReprError;
using robopt::reprojection::LocalRefEuclideanReprError;

namespace robopt {

class LocalEuclideanErrorTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  typedef aslam::EquidistantDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;
  typedef LocalEuclideanReprError<CameraType, DistortionType> ErrorTerm;
  typedef LocalRefEuclideanReprError<CameraType, DistortionType> ErrorTermRef;
  
  virtual void SetUp() {
    zero_position_.setZero();
    unit_quaternion_.setIdentity();
    
    // Camera Parameters obtained by the V4RL VI-Sensor
    distortion_params_ = Eigen::VectorXd(4);
    distortion_params_ << 0.0007963927229700795, 0.0185916918332955, 
        -0.03878055672150203, 0.02734328895604988;
    fu_ = 469.76997764418866;
    fv_ = 468.29933542369247;
    cu_ = 373.5113658284214;
    cv_ = 246.72520189124873;
    res_u_ = 752.0;
    res_v_ = 480.0;
    
    pixel_sigma_ = 1.5;
    
    dummy_7d_0_ << 0, 0, 0, 1, 0, 0, 0;
    dummy_7d_1_ << 0, 0, 0, 1, 0, 0, 0;
  }

  void constructCamera() {
    Eigen::VectorXd distortion_parameters(4);
    distortion_parameters << distortion_params_;
    aslam::Distortion::UniquePtr distortion(
        new DistortionType(distortion_parameters));

    Eigen::VectorXd intrinsics(4);
    intrinsics << fu_, fv_, cu_, cv_;

    camera_.reset(new CameraType(intrinsics, res_u_, res_v_, distortion));
  }

  ceres::Problem problem_;
  ceres::Solver::Summary summary_;
  ceres::Solver::Options options_;

  std::shared_ptr<CameraType> camera_;

  Eigen::Vector3d zero_position_;
  Eigen::Quaterniond unit_quaternion_;

  Eigen::VectorXd distortion_params_;
  double fu_, fv_;
  double cu_, cv_;
  double res_u_, res_v_;
  double pixel_sigma_;

  // Ordering is [orientation position] -> [xyzw xyz].
  Eigen::Matrix<double, 7, 1> dummy_7d_0_;
  Eigen::Matrix<double, 7, 1> dummy_7d_1_;
};


TEST_F(LocalEuclideanErrorTerms, Jacobians) {
  // Create an extrinsic transformation
  const Eigen::Vector3d p_S_C(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_S_C(0.5, -0.2, 0.4, 0.5);
  q_S_C.normalize();
  const Eigen::Matrix3d R_S_C = q_S_C.toRotationMatrix();

  // Create a Keyframe Pose
  const Eigen::Vector3d p_W_S(4.0, 3.2, -1.5);
  Eigen::Quaterniond q_W_S(-0.3, 0.5, 0.9, -0.2);
  q_W_S.normalize();
  const Eigen::Matrix3d R_W_S = q_W_S.toRotationMatrix();

  // Create a Reference Pose
  const Eigen::Vector3d p_W_R(2.0, -1.4, 0.5);
  Eigen::Quaterniond q_W_R(0.6, -0.1, 0.4, 0.3);
  q_W_R.normalize();
  const Eigen::Matrix3d R_W_R = q_W_R.toRotationMatrix();

  // Create a new camera
  constructCamera();

  // Create a landmark
  const Eigen::Vector3d l_C(0.3, 0.4, 4.2);
  const Eigen::Vector3d l_S = R_S_C*l_C + p_S_C;
  const Eigen::Vector3d l_W = R_W_S*l_S + p_W_S;
  const Eigen::Vector3d l_R = R_W_R.transpose() * (l_W - p_W_R);

  // Create a measurement
  Eigen::Vector2d projection;
  const aslam::ProjectionResult projection_result =
      camera_->project3(l_C, &projection);
  projection += Eigen::Vector2d(-1.0, 0.4);

  // Create the raw pointers to the data
  double** parameters_nom = new double*[6];
  double** parameters_dist = new double*[6];
  parameters_nom[0] = new double[defs::visual::kPoseBlockSize]{
      q_W_S.x(), q_W_S.y(), q_W_S.z(), q_W_S.w(),
      p_W_S[0], p_W_S[1], p_W_S[2]};
  parameters_dist[0] = new double[defs::visual::kPoseBlockSize]{
    q_W_S.x(), q_W_S.y(), q_W_S.z(), q_W_S.w(),
    p_W_S[0], p_W_S[1], p_W_S[2]};
  parameters_nom[1] = new double[defs::visual::kPoseBlockSize]{
      q_S_C.x(), q_S_C.y(), q_S_C.z(), q_S_C.w(),
      p_S_C[0], p_S_C[1], p_S_C[2]};
  parameters_dist[1] = new double[defs::visual::kPoseBlockSize]{
      q_S_C.x(), q_S_C.y(), q_S_C.z(), q_S_C.w(),
      p_S_C[0], p_S_C[1], p_S_C[2]};
  parameters_nom[2] = new double[defs::visual::kPoseBlockSize]{
      q_W_R.x(), q_W_R.y(), q_W_R.z(), q_W_R.w(),
      p_W_R[0], p_W_R[1], p_W_R[2]};
  parameters_dist[2] = new double[defs::visual::kPoseBlockSize]{
      q_W_R.x(), q_W_R.y(), q_W_R.z(), q_W_R.w(),
      p_W_R[0], p_W_R[1], p_W_R[2]};
  parameters_nom[3] = new double[defs::visual::kPositionBlockSize]{
      l_R[0], l_R[1], l_R[2]};
  parameters_dist[3] = new double[defs::visual::kPositionBlockSize]{
      l_R[0], l_R[1], l_R[2]};
  parameters_nom[4] = new double[CameraType::parameterCount()];
  parameters_dist[4] = new double[CameraType::parameterCount()];
  for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
    parameters_nom[4][i] = camera_->getParameters()[i];
    parameters_dist[4][i] = camera_->getParameters()[i];
  }
  parameters_nom[5] = new double[DistortionType::parameterCount()];
  parameters_dist[5] = new double[DistortionType::parameterCount()];
  for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
    parameters_nom[5][i] = camera_->getDistortion().getParameters()[i];
    parameters_dist[5][i] = camera_->getDistortion().getParameters()[i];
  }

  // Create the analytical jacobians
  double** jacobians_analytical = new double*[5];
  jacobians_analytical[0] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  jacobians_analytical[1] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  jacobians_analytical[2] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  jacobians_analytical[3] =
      new double[defs::visual::kResidualSize*defs::visual::kPositionBlockSize];
  jacobians_analytical[4] =
      new double[defs::visual::kResidualSize*CameraType::parameterCount()];
  jacobians_analytical[5] =
      new double[defs::visual::kResidualSize*DistortionType::parameterCount()];
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
      J_keypoint_wrt_T_W_S(jacobians_analytical[0]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
      J_keypoint_wrt_T_S_C(jacobians_analytical[1]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
      J_keypoint_wrt_T_W_R(jacobians_analytical[2]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            defs::visual::kPositionBlockSize, Eigen::RowMajor>>
      J_keypoint_wrt_l_R(jacobians_analytical[3]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            CameraType::parameterCount(), Eigen::RowMajor>>
      J_keypoint_wrt_intrinsics(jacobians_analytical[4]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            DistortionType::parameterCount(), Eigen::RowMajor>>
      J_keypoint_wrt_distortion(jacobians_analytical[5]);

  // Compute the analytical derivatives
  ErrorTerm error_term(projection, pixel_sigma_, camera_.get());
  double* residuals_nom = new double[defs::visual::kResidualSize];
  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

  // Prepare the numerical jacobians
  double* residuals_dist = new double[defs::visual::kResidualSize];
  Eigen::Matrix<double, defs::visual::kResidualSize,
               defs::visual::kPoseBlockSize>
      J_keypoint_wrt_T_W_S_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
               defs::visual::kPoseBlockSize>
      J_keypoint_wrt_T_S_C_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
               defs::visual::kPoseBlockSize>
      J_keypoint_wrt_T_W_R_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
               defs::visual::kPositionBlockSize>
      J_keypoint_wrt_l_R_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
               CameraType::parameterCount()>
      J_keypoint_wrt_intrinsics_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
               DistortionType::parameterCount()>
      J_keypoint_wrt_distortion_num;
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
      residual_map_nom(residuals_nom);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
      residual_map_dist(residuals_dist);
  const double delta = 1e-7;
  Eigen::Matrix<double, defs::visual::kResidualSize, 1>  difference;
  Eigen::Map<Eigen::Quaterniond> q_W_S_dist(parameters_dist[0]);
  Eigen::Map<Eigen::Vector3d> p_W_S_dist(parameters_dist[0] +
      defs::visual::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q_S_C_dist(parameters_dist[1]);
  Eigen::Map<Eigen::Vector3d> p_S_C_dist(parameters_dist[1] +
      defs::visual::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q_W_R_dist(parameters_dist[2]);
  Eigen::Map<Eigen::Vector3d> p_W_R_dist(parameters_dist[2] +
      defs::visual::kOrientationBlockSize);
  Eigen::Map<Eigen::Vector3d> l_R_dist(parameters_dist[3]);

  // Jacobian for T_W_S
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = Eigen::Quaterniond(q_W_S.w() + dist[3],
        q_W_S.x() + dist[0], q_W_S.y() + dist[1], q_W_S.z() + dist[2]);
    q_W_S_dist.normalize();
    p_W_S_dist = p_W_S + dist.tail<3>();

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for T_W_R
    q_W_R_dist = q_W_R;
    p_W_R_dist = p_W_R;

    // Map the parameters for l_C
    l_R_dist = l_R;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[5][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_W_S_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for T_S_C
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = Eigen::Quaterniond(q_S_C.w() + dist[3],
        q_S_C.x() + dist[0], q_S_C.y() + dist[1], q_S_C.z() + dist[2]);
    q_S_C_dist.normalize();
    p_S_C_dist = p_S_C + dist.tail<3>();

    // Map the parameters for T_W_R
    q_W_R_dist = q_W_R;
    p_W_R_dist = p_W_R;

    // Map the parameters for l_C
    l_R_dist = l_R;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[5][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_S_C_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for T_W_R
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for T_W_R
    q_W_R_dist = Eigen::Quaterniond(q_W_R.w() + dist[3],
        q_W_R.x() + dist[0], q_W_R.y() + dist[1], q_W_R.z() + dist[2]);
    q_W_R_dist.normalize();
    p_W_R_dist = p_W_R + dist.tail<3>();

    // Map the parameters for l_C
    l_R_dist = l_R;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[5][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_W_R_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for l_R
  for (size_t i = 0; i < defs::visual::kPositionBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPositionBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for T_W_R
    q_W_R_dist = q_W_R;
    p_W_R_dist = p_W_R;

    // Map the parameters for l_C
    l_R_dist = l_R + dist;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[5][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_l_R_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for intrinsics
  for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
    Eigen::VectorXd dist(CameraType::parameterCount());
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for T_W_R
    q_W_R_dist = q_W_R;
    p_W_R_dist = p_W_R;

    // Map the parameters for l_C
    l_R_dist = l_R;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getParameters()[i] + dist[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[5][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_intrinsics_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for Distortion
  for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
    Eigen::VectorXd dist(DistortionType::parameterCount());
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for T_W_R
    q_W_R_dist = q_W_R;
    p_W_R_dist = p_W_R;

    // Map the parameters for l_C
    l_R_dist = l_R;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[5][i] = camera_->getDistortion().getParameters()[i] +
          dist[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_distortion_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_W_S,
                          J_keypoint_wrt_T_W_S_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_S_C,
                          J_keypoint_wrt_T_S_C_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_W_R,
                            J_keypoint_wrt_T_W_R_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_l_R,
                          J_keypoint_wrt_l_R_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_W_S,
                          J_keypoint_wrt_T_W_S_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_intrinsics,
                          J_keypoint_wrt_intrinsics_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_distortion,
                          J_keypoint_wrt_distortion_num, 1e-4);
}

TEST_F(LocalEuclideanErrorTerms, JacobiansRef) {
  // Create an extrinsic transformation
  const Eigen::Vector3d p_S_C(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_S_C(0.5, -0.2, 0.4, 0.5);
  q_S_C.normalize();
  const Eigen::Matrix3d R_S_C = q_S_C.toRotationMatrix();

  // Create a Keyframe Pose
  const Eigen::Vector3d p_W_S(4.0, 3.2, -1.5);
  Eigen::Quaterniond q_W_S(-0.3, 0.5, 0.9, -0.2);
  q_W_S.normalize();
  const Eigen::Matrix3d R_W_S = q_W_S.toRotationMatrix();

  // Create a new camera
  constructCamera();

  // Create a landmark
  const Eigen::Vector3d l_C(0.3, 0.4, 4.2);
  const Eigen::Vector3d l_S = R_S_C*l_C + p_S_C;
  const Eigen::Vector3d l_W = R_W_S*l_S + p_W_S;

  // Create a measurement
  Eigen::Vector2d projection;
  const aslam::ProjectionResult projection_result =
      camera_->project3(l_C, &projection);
  projection += Eigen::Vector2d(-1.0, 0.4);

  // Create the raw pointers to the data
  double** parameters_nom = new double*[5];
  double** parameters_dist = new double*[5];
  parameters_nom[0] = new double[defs::visual::kPoseBlockSize]{
      q_W_S.x(), q_W_S.y(), q_W_S.z(), q_W_S.w(),
      p_W_S[0], p_W_S[1], p_W_S[2]};
  parameters_dist[0] = new double[defs::visual::kPoseBlockSize]{
      q_W_S.x(), q_W_S.y(), q_W_S.z(), q_W_S.w(),
      p_W_S[0], p_W_S[1], p_W_S[2]};
  parameters_nom[1] = new double[defs::visual::kPoseBlockSize]{
      q_S_C.x(), q_S_C.y(), q_S_C.z(), q_S_C.w(),
      p_S_C[0], p_S_C[1], p_S_C[2]};
  parameters_dist[1] = new double[defs::visual::kPoseBlockSize]{
      q_S_C.x(), q_S_C.y(), q_S_C.z(), q_S_C.w(),
      p_S_C[0], p_S_C[1], p_S_C[2]};
  parameters_nom[2] = new double[defs::visual::kPositionBlockSize]{
      l_S[0], l_S[1], l_S[2]};
  parameters_dist[2] = new double[defs::visual::kPositionBlockSize]{
      l_S[0], l_S[1], l_S[2]};
  parameters_nom[3] = new double[CameraType::parameterCount()];
  parameters_dist[3] = new double[CameraType::parameterCount()];
  for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
    parameters_nom[3][i] = camera_->getParameters()[i];
    parameters_dist[3][i] = camera_->getParameters()[i];
  }
  parameters_nom[4] = new double[DistortionType::parameterCount()];
  parameters_dist[4] = new double[DistortionType::parameterCount()];
  for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
    parameters_nom[4][i] = camera_->getDistortion().getParameters()[i];
    parameters_dist[4][i] = camera_->getDistortion().getParameters()[i];
  }

  // Create the analytical jacobians
  double** jacobians_analytical = new double*[5];
  jacobians_analytical[0] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  jacobians_analytical[1] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  jacobians_analytical[2] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  jacobians_analytical[3] =
      new double[defs::visual::kResidualSize*CameraType::parameterCount()];
  jacobians_analytical[4] =
      new double[defs::visual::kResidualSize*DistortionType::parameterCount()];
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
  defs::visual::kPoseBlockSize, Eigen::RowMajor>>
  J_keypoint_wrt_T_W_S(jacobians_analytical[0]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
  defs::visual::kPoseBlockSize, Eigen::RowMajor>>
  J_keypoint_wrt_T_S_C(jacobians_analytical[1]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
  defs::visual::kPositionBlockSize, Eigen::RowMajor>>
  J_keypoint_wrt_l_S(jacobians_analytical[2]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
  CameraType::parameterCount(), Eigen::RowMajor>>
    J_keypoint_wrt_intrinsics(jacobians_analytical[3]);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
  DistortionType::parameterCount(), Eigen::RowMajor>>
    J_keypoint_wrt_distortion(jacobians_analytical[4]);

  // Compute the analytical derivatives
  ErrorTermRef error_term(projection, pixel_sigma_, camera_.get());
  double* residuals_nom = new double[defs::visual::kResidualSize];
  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

  // Prepare the numerical jacobians
  double* residuals_dist = new double[defs::visual::kResidualSize];
  Eigen::Matrix<double, defs::visual::kResidualSize,
  defs::visual::kPoseBlockSize>
  J_keypoint_wrt_T_W_S_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
  defs::visual::kPoseBlockSize>
  J_keypoint_wrt_T_S_C_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
  defs::visual::kPositionBlockSize>
  J_keypoint_wrt_l_S_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
  CameraType::parameterCount()>
  J_keypoint_wrt_intrinsics_num;
  Eigen::Matrix<double, defs::visual::kResidualSize,
  DistortionType::parameterCount()>
  J_keypoint_wrt_distortion_num;
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
  residual_map_nom(residuals_nom);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
  residual_map_dist(residuals_dist);
  const double delta = 1e-7;
  Eigen::Matrix<double, defs::visual::kResidualSize, 1>  difference;
  Eigen::Map<Eigen::Quaterniond> q_W_S_dist(parameters_dist[0]);
  Eigen::Map<Eigen::Vector3d> p_W_S_dist(parameters_dist[0] +
  defs::visual::kOrientationBlockSize);
  Eigen::Map<Eigen::Quaterniond> q_S_C_dist(parameters_dist[1]);
  Eigen::Map<Eigen::Vector3d> p_S_C_dist(parameters_dist[1] +
  defs::visual::kOrientationBlockSize);
  Eigen::Map<Eigen::Vector3d> l_S_dist(parameters_dist[2]);

  // Jacobian for T_W_S
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = Eigen::Quaterniond(q_W_S.w() + dist[3],
        q_W_S.x() + dist[0], q_W_S.y() + dist[1], q_W_S.z() + dist[2]);
    q_W_S_dist.normalize();
    p_W_S_dist = p_W_S + dist.tail<3>();

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for l_S
    l_S_dist = l_S;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[3][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_W_S_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for T_S_C
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = Eigen::Quaterniond(q_S_C.w() + dist[3],
        q_S_C.x() + dist[0], q_S_C.y() + dist[1], q_S_C.z() + dist[2]);
    q_S_C_dist.normalize();
    p_S_C_dist = p_S_C + dist.tail<3>();

    // Map the parameters for l_S
    l_S_dist = l_S;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[3][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_S_C_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for l_S
  for (size_t i = 0; i < defs::visual::kPositionBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPositionBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for l_S
    l_S_dist = l_S + dist;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[3][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_l_S_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for intrinsics
  for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
    Eigen::VectorXd dist(CameraType::parameterCount());
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for l_C
    l_S_dist = l_S;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[3][i] = camera_->getParameters()[i] + dist[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getDistortion().getParameters()[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_intrinsics_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Jacobian for Distortion
  for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
    Eigen::VectorXd dist(DistortionType::parameterCount());
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_W_S_dist = q_W_S;
    p_W_S_dist = p_W_S;

    // Map the parameters for T_S_C
    q_S_C_dist = q_S_C;
    p_S_C_dist = p_S_C;

    // Map the parameters for l_C
    l_S_dist = l_S;

    // Map the parameters for intrinsics
    for (size_t i = 0; i < CameraType::parameterCount(); ++i) {
      parameters_dist[3][i] = camera_->getParameters()[i];
    }
    for (size_t i = 0; i < DistortionType::parameterCount(); ++i) {
      parameters_dist[4][i] = camera_->getDistortion().getParameters()[i] +
          dist[i];
    }

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_distortion_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_W_S,
                          J_keypoint_wrt_T_W_S_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_S_C,
                          J_keypoint_wrt_T_S_C_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_l_S,
                          J_keypoint_wrt_l_S_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_W_S,
                          J_keypoint_wrt_T_W_S_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_intrinsics,
                          J_keypoint_wrt_intrinsics_num, 1e-4);
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_distortion,
                          J_keypoint_wrt_distortion_num, 1e-4);
}

}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
