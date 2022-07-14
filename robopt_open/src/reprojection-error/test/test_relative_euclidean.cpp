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
#include <reprojection-error/relative-euclidean.h>

using robopt::reprojection::RelativeEuclideanReprError;

namespace robopt {

class RelativeEuclideanErrorTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  typedef aslam::EquidistantDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;
  typedef RelativeEuclideanReprError<CameraType, DistortionType> ErrorTerm;
  
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


TEST_F(RelativeEuclideanErrorTerms, JacobiansNormal) {
  // Create a relative transformation
  const Eigen::Vector3d p_A_B(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_A_B(0.5, -0.2, 0.4, 0.5);
  q_A_B.normalize();
  const Eigen::Matrix3d R_A_B = q_A_B.toRotationMatrix();

  // Create a new camera
  constructCamera();

  // Create a suitable point
  Eigen::Vector2d kp (200, 300);
  Eigen::Vector3d point_C;
  camera_->backProject3(kp, &point_C);
  point_C = point_C * 4.8;
  Eigen::Vector2d measurement = kp + Eigen::Vector2d(0.4, 0.5);
  Eigen::Vector3d point_R = R_A_B.transpose() * (point_C - p_A_B);

  // Create the raw pointers to the data
  double** parameters_nom = new double*[1];
  double** parameters_dist = new double*[1];
  parameters_nom[0] = new double[defs::visual::kPoseBlockSize]{
      q_A_B.x(), q_A_B.y(), q_A_B.z(), q_A_B.w(),
      p_A_B[0], p_A_B[1], p_A_B[2]};
  parameters_dist[0] = new double[defs::visual::kPoseBlockSize]{
    q_A_B.x(), q_A_B.y(), q_A_B.z(), q_A_B.w(),
    p_A_B[0], p_A_B[1], p_A_B[2]};

  // Create the analytical jacobians
  double** jacobians_analytical = new double*[1];
  jacobians_analytical[0] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
      J_keypoint_wrt_T_A_B(jacobians_analytical[0]);

  // Compute the analytical derivatives
  ErrorTerm error_term(measurement, pixel_sigma_, camera_.get(),
                       point_R, defs::visual::RelativeProjectionType::kNormal);
  double* residuals_nom = new double[defs::visual::kResidualSize];
  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

  // Prepare the numerical jacobians
  double* residuals_dist = new double[defs::visual::kResidualSize];
  Eigen::Matrix<double, defs::visual::kResidualSize,
               defs::visual::kPoseBlockSize>
      J_keypoint_wrt_T_A_B_num;
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
  residual_map_nom(residuals_nom);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
  residual_map_dist(residuals_dist);
  const double delta = 1e-7;
  Eigen::Matrix<double, defs::visual::kResidualSize, 1>  difference;
  Eigen::Map<Eigen::Quaterniond> q_A_B_dist(parameters_dist[0]);
  Eigen::Map<Eigen::Vector3d> p_A_B_dist(parameters_dist[0] +
      defs::visual::kOrientationBlockSize);

  // Jacobian for T_W_S
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_A_B_dist = Eigen::Quaterniond(q_A_B.w() + dist[3],
        q_A_B.x() + dist[0], q_A_B.y() + dist[1], q_A_B.z() + dist[2]);
    q_A_B_dist.normalize();
    p_A_B_dist = p_A_B + dist.tail<3>();

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_A_B_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_A_B,
                          J_keypoint_wrt_T_A_B_num, 1e-4);
}

TEST_F(RelativeEuclideanErrorTerms, JacobiansInverse) {
  // Create a relative transformation
  const Eigen::Vector3d p_A_B(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_A_B(0.5, -0.2, 0.4, 0.5);
  q_A_B.normalize();
  const Eigen::Matrix3d R_A_B = q_A_B.toRotationMatrix();

  // Create a new camera
  constructCamera();

  // Create a suitable point
  Eigen::Vector2d kp (200, 300);
  Eigen::Vector3d point_C;
  camera_->backProject3(kp, &point_C);
  point_C = point_C * 4.8;
  Eigen::Vector2d measurement = kp + Eigen::Vector2d(0.4, 0.5);
  Eigen::Vector3d point_R = R_A_B * point_C + p_A_B;

  // Create the raw pointers to the data
  double** parameters_nom = new double*[1];
  double** parameters_dist = new double*[1];
  parameters_nom[0] = new double[defs::visual::kPoseBlockSize]{
      q_A_B.x(), q_A_B.y(), q_A_B.z(), q_A_B.w(),
      p_A_B[0], p_A_B[1], p_A_B[2]};
  parameters_dist[0] = new double[defs::visual::kPoseBlockSize]{
    q_A_B.x(), q_A_B.y(), q_A_B.z(), q_A_B.w(),
    p_A_B[0], p_A_B[1], p_A_B[2]};

  // Create the analytical jacobians
  double** jacobians_analytical = new double*[1];
  jacobians_analytical[0] =
      new double[defs::visual::kResidualSize*defs::visual::kPoseBlockSize];
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize,
            defs::visual::kPoseBlockSize, Eigen::RowMajor>>
      J_keypoint_wrt_T_A_B(jacobians_analytical[0]);

  // Compute the analytical derivatives
  ErrorTerm error_term(measurement, pixel_sigma_, camera_.get(),
                       point_R, defs::visual::RelativeProjectionType::kInverse);
  double* residuals_nom = new double[defs::visual::kResidualSize];
  error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);

  // Prepare the numerical jacobians
  double* residuals_dist = new double[defs::visual::kResidualSize];
  Eigen::Matrix<double, defs::visual::kResidualSize,
               defs::visual::kPoseBlockSize>
      J_keypoint_wrt_T_A_B_num;
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
  residual_map_nom(residuals_nom);
  Eigen::Map<Eigen::Matrix<double, defs::visual::kResidualSize, 1>>
  residual_map_dist(residuals_dist);
  const double delta = 1e-7;
  Eigen::Matrix<double, defs::visual::kResidualSize, 1>  difference;
  Eigen::Map<Eigen::Quaterniond> q_A_B_dist(parameters_dist[0]);
  Eigen::Map<Eigen::Vector3d> p_A_B_dist(parameters_dist[0] +
      defs::visual::kOrientationBlockSize);

  // Jacobian for T_W_S
  for (size_t i = 0; i < defs::visual::kPoseBlockSize; ++i) {
    Eigen::VectorXd dist(defs::visual::kPoseBlockSize);
    dist.setZero();
    dist[i] = delta;

    // Map the parameters for T_W_S
    q_A_B_dist = Eigen::Quaterniond(q_A_B.w() + dist[3],
        q_A_B.x() + dist[0], q_A_B.y() + dist[1], q_A_B.z() + dist[2]);
    q_A_B_dist.normalize();
    p_A_B_dist = p_A_B + dist.tail<3>();

    // Compute the numerical difference
    error_term.Evaluate(parameters_dist, residuals_dist, NULL);
    difference = residual_map_dist - residual_map_nom;
    J_keypoint_wrt_T_A_B_num.block<defs::visual::kResidualSize, 1>(0, i) =
        difference/delta;
  }

  // Carry out the checks
  CHECK_EIGEN_MATRIX_NEAR(J_keypoint_wrt_T_A_B,
                          J_keypoint_wrt_T_A_B_num, 1e-4);
}

}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
