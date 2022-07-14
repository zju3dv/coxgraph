#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/distortion-equidistant.h>

#include <reprojection-error/triangulation.h>

using robopt::reprojection::TriangulationStatus;

namespace robopt {

class TriangulationTest : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TriangulationTest() {
    srand((unsigned int) time(0));
  }

protected:
  typedef aslam::EquidistantDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;
  
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


TEST_F(TriangulationTest, SVDTriang) {
  // Create a landmark
  Eigen::Vector3d l_W(1.5, 3.3, 9.6);

  // Create a fixed extrinsics transformation
  const Eigen::Vector3d p_S_C(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_S_C(0.5, -0.2, 0.4, 0.5);
  q_S_C.normalize();

  // Create disturbed poses
  const size_t num_poses = 10;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      T_W_Si;
  T_W_Si.reserve(num_poses);
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      T_S_Ci;
  T_S_Ci.reserve(num_poses);
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      meas_normalized_i;
  meas_normalized_i.reserve(num_poses);
  for (size_t i = 0; i < num_poses; ++i) {
    Eigen::Matrix4d T_S_C = Eigen::Matrix4d::Identity();
    T_S_C.block<3,3>(0, 0) = q_S_C.toRotationMatrix();
    T_S_C.block<3,1>(0, 3) = p_S_C;
    T_S_Ci.push_back(T_S_C);

    Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
    T_W_S.block<3,1>(0, 3) = Eigen::Vector3d(1.0, 3, 0);
    Eigen::Matrix4d T_dist = Eigen::Matrix4d::Identity();
    T_dist(0, 3) = Eigen::internal::random(-1.5,1.5);
    T_dist(1, 3) = Eigen::internal::random(-1.5,1.5);
    T_dist(2, 3) = Eigen::internal::random(-1.5,1.5);
    Eigen::Vector3d rand_rot(
          Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1));
    Eigen::Quaterniond q_dist(
          1.0, Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1), Eigen::internal::random(-0.1,0.1));
    T_dist.block<3,3>(0, 0) = q_dist.toRotationMatrix();
    T_W_S = T_W_S * T_dist;
    T_W_Si.push_back(T_W_S);

    Eigen::Matrix4d T_C_W = T_S_C.inverse() * T_W_S.inverse();
    Eigen::Vector3d l_C = T_C_W.block<3,3>(0, 0) * l_W +
        T_C_W.block<3,1>(0, 3);
    Eigen::Vector2d meas_norm = l_C.head<2>() / l_C(2);
    meas_normalized_i.push_back(meas_norm);
  }

  Eigen::Vector3d triangulation;
  TriangulationStatus status = reprojection::svdTriangulation(
        T_W_Si, T_S_Ci, meas_normalized_i, &triangulation);
  CHECK_EIGEN_MATRIX_NEAR(triangulation, l_W, 1e-8);
}

TEST_F(TriangulationTest, NonlinTriangAcc) {
  // Create a landmark
  Eigen::Vector3d l_W(1.5, 3.3, 9.6);

  // Create a fixed extrinsics transformation
  const Eigen::Vector3d p_S_C(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_S_C(0.5, -0.2, 0.4, 0.5);
  q_S_C.normalize();

  // Create disturbed poses
  const size_t num_poses = 10;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      T_W_Si;
  T_W_Si.reserve(num_poses);
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      T_S_Ci;
  T_S_Ci.reserve(num_poses);
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      meas_normalized_i;
  meas_normalized_i.reserve(num_poses);
  std::vector<double> sqrt_info;
  sqrt_info.reserve(num_poses);
  for (size_t i = 0; i < num_poses; ++i) {
    Eigen::Matrix4d T_S_C = Eigen::Matrix4d::Identity();
    T_S_C.block<3,3>(0, 0) = q_S_C.toRotationMatrix();
    T_S_C.block<3,1>(0, 3) = p_S_C;
    T_S_Ci.push_back(T_S_C);

    Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
    T_W_S.block<3,1>(0, 3) = Eigen::Vector3d(1.0, 3, 0);
    Eigen::Matrix4d T_dist = Eigen::Matrix4d::Identity();
    T_dist(0, 3) = Eigen::internal::random(-1.5,1.5);
    T_dist(1, 3) = Eigen::internal::random(-1.5,1.5);
    T_dist(2, 3) = Eigen::internal::random(-1.5,1.5);
    Eigen::Vector3d rand_rot(
          Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1));
    Eigen::Quaterniond q_dist(
          1.0, Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1), Eigen::internal::random(-0.1,0.1));
    T_dist.block<3,3>(0, 0) = q_dist.toRotationMatrix();
    T_W_S = T_W_S * T_dist;
    T_W_Si.push_back(T_W_S);

    Eigen::Matrix4d T_C_W = T_S_C.inverse() * T_W_S.inverse();
    Eigen::Vector3d l_C = T_C_W.block<3,3>(0, 0) * l_W +
        T_C_W.block<3,1>(0, 3);
    Eigen::Vector2d meas_norm = l_C.head<2>() / l_C(2);
    meas_normalized_i.push_back(meas_norm);
    sqrt_info.push_back(std::sqrt(450.0/1.0));
  }

  Eigen::Vector3d triangulation;
  triangulation = l_W + Eigen::Vector3d(
        Eigen::internal::random(-0.4, 0.4),
        Eigen::internal::random(-0.4, 0.4),
        Eigen::internal::random(-0.5, 0.5));
  std::vector<bool> outliers;
  outliers.resize(num_poses, false);
  TriangulationStatus status = reprojection::nonlinearTriangulation(
        T_W_Si, T_S_Ci, meas_normalized_i, sqrt_info, &triangulation, outliers,
        1.0);

  CHECK_EIGEN_MATRIX_NEAR(triangulation, l_W, 1e-8);
}

TEST_F(TriangulationTest, NonlinTriangOutlier) {
  // Create a landmark
  Eigen::Vector3d l_W(1.5, 3.3, 9.6);

  // Create a fixed extrinsics transformation
  const Eigen::Vector3d p_S_C(0.5, -0.3, 1.0);
  Eigen::Quaterniond q_S_C(0.5, -0.2, 0.4, 0.5);
  q_S_C.normalize();

  // Create disturbed poses
  const size_t num_poses = 10;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      T_W_Si;
  T_W_Si.reserve(num_poses);
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      T_S_Ci;
  T_S_Ci.reserve(num_poses);
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      meas_normalized_i;
  meas_normalized_i.reserve(num_poses);
  std::vector<double> sqrt_info;
  sqrt_info.reserve(num_poses);
  std::vector<bool> outlier_gt;
  outlier_gt.resize(num_poses, false);
  for (size_t i = 0; i < num_poses; ++i) {
    Eigen::Matrix4d T_S_C = Eigen::Matrix4d::Identity();
    T_S_C.block<3,3>(0, 0) = q_S_C.toRotationMatrix();
    T_S_C.block<3,1>(0, 3) = p_S_C;
    T_S_Ci.push_back(T_S_C);

    Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
    T_W_S.block<3,1>(0, 3) = Eigen::Vector3d(1.0, 3, 0);
    Eigen::Matrix4d T_dist = Eigen::Matrix4d::Identity();
    T_dist(0, 3) = Eigen::internal::random(-1.5,1.5);
    T_dist(1, 3) = Eigen::internal::random(-1.5,1.5);
    T_dist(2, 3) = Eigen::internal::random(-1.5,1.5);
    Eigen::Vector3d rand_rot(
          Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1));
    Eigen::Quaterniond q_dist(
          1.0, Eigen::internal::random(-0.1,0.1),
          Eigen::internal::random(-0.1,0.1), Eigen::internal::random(-0.1,0.1));
    T_dist.block<3,3>(0, 0) = q_dist.toRotationMatrix();
    T_W_S = T_W_S * T_dist;
    T_W_Si.push_back(T_W_S);

    Eigen::Matrix4d T_C_W = T_S_C.inverse() * T_W_S.inverse();
    Eigen::Vector3d l_C = T_C_W.block<3,3>(0, 0) * l_W +
        T_C_W.block<3,1>(0, 3);
    Eigen::Vector2d meas_norm = l_C.head<2>() / l_C(2);
    if (i == 3 || i == 7) {
      outlier_gt[i] = true;
      meas_norm += Eigen::Vector2d(
            10.0/450.0,
            -10.0/450.0);
    }
    meas_normalized_i.push_back(meas_norm);
    sqrt_info.push_back(std::sqrt(450.0/1.0));
  }

  Eigen::Vector3d triangulation;
  triangulation = l_W + Eigen::Vector3d(
        Eigen::internal::random(-0.4, 0.4),
        Eigen::internal::random(-0.4, 0.4),
        Eigen::internal::random(-0.4, 0.4));
  std::vector<bool> outliers;
  outliers.resize(num_poses, false);

  TriangulationStatus status = reprojection::nonlinearTriangulation(
        T_W_Si, T_S_Ci, meas_normalized_i, sqrt_info, &triangulation, outliers,
        1.0);

  // Check if outliers are correctly identified
  for (size_t i = 0; i < num_poses; ++i) {
    CHECK_EQ(outliers[i], outlier_gt[i]);
  }
}

}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
