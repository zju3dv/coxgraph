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
#include <reprojection-error/smart-projection.h>

using robopt::reprojection::SmartProjectionError;

namespace robopt {

class SmartProjectionErrorTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SmartProjectionErrorTerms() {
    srand((unsigned int) time(0));
  }

protected:
  typedef aslam::EquidistantDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;
  typedef SmartProjectionError<CameraType, DistortionType> ErrorTerm;
  
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

    num_cams_ = 15;
    num_landmarks_ = 100;
    landmark_distance_ = 10.0;
    trajectory_length_ = 15.0;
  }

  struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector<size_t> ids;
    std::vector<Eigen::Vector2d> observations;
    Eigen::Quaterniond q_W_S;
    Eigen::Quaterniond q_W_S_init;
    Eigen::Quaterniond q_S_C;
    Eigen::Quaterniond q_S_C_init;
    Eigen::Vector3d p_W_S;
    Eigen::Vector3d p_W_S_init;
    Eigen::Vector3d p_S_C;
    Eigen::Vector3d p_S_C_init;
    size_t frame_id;
  };

  void constructCamera() {
    Eigen::VectorXd distortion_parameters(4);
    distortion_parameters << distortion_params_;
    aslam::Distortion::UniquePtr distortion(
        new DistortionType(distortion_parameters));

    Eigen::VectorXd intrinsics(4);
    intrinsics << fu_, fv_, cu_, cv_;

    camera_.reset(new CameraType(intrinsics, res_u_, res_v_, distortion));
  }

  void constructProblem() {
    Eigen::Quaterniond q_S_C;
    q_S_C.setIdentity();
    Eigen::Vector3d p_S_C(0.5, 0.4,-0.4);
    //p_S_C.setZero();

    // Create the keyframes
    frames_.reserve(num_cams_);
    for (size_t i = 0; i < num_cams_; ++i) {
      Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
      T_W_S(0, 3) = i * trajectory_length_/((double) num_cams_);
      T_W_S(1, 3) = 0.0;
      T_W_S(2, 3) = 0.0;
      Eigen::Vector3d axis = Eigen::Vector3d::Random() * 0.1;
      Eigen::Quaterniond q_dist(Eigen::AngleAxisd(axis.norm(),
          axis.normalized()));
      Eigen::Vector3d t_dist = Eigen::Vector3d::Random() * 0.4;
      Eigen::Matrix4d T_dist = Eigen::Matrix4d::Identity();
      T_dist.block<3,3>(0, 0) = q_dist.toRotationMatrix();
      T_dist.block<3,1>(0, 3) = t_dist;
      T_W_S = T_W_S * T_dist;
      Frame frame;
      frame.p_S_C = p_S_C;
      frame.q_S_C = q_S_C;
      frame.p_W_S = T_W_S.block<3,1>(0, 3);
      frame.q_W_S = Eigen::Quaterniond(T_W_S.block<3,3>(0, 0));
      frame.frame_id = i;

      // Create the disturbed initial poses
      axis = Eigen::Vector3d::Random() * 0.1;
      q_dist = Eigen::Quaterniond(Eigen::AngleAxisd(axis.norm(),
                axis.normalized()));
      t_dist = Eigen::Vector3d::Random() * 0.1;
      T_dist.setIdentity();
      if (i > 1) {
        T_dist.block<3,3>(0, 0) = q_dist.toRotationMatrix();
        T_dist.block<3,1>(0, 3) = t_dist;
      }
      T_W_S = T_W_S * T_dist;

      frame.q_S_C_init = q_S_C;
      frame.p_S_C_init = p_S_C;
      frame.q_W_S_init = Eigen::Quaterniond(T_W_S.block<3,3>(0, 0));
      frame.p_W_S_init = T_W_S.block<3,1>(0, 3);

      frames_.push_back(frame);
    }

    // Create the landmarks
    landmarks_.reserve(num_landmarks_);
    landmark_ids_.reserve(num_landmarks_);
    for (size_t i = 0; i < num_landmarks_; ++i) {
      Eigen::Vector3d tmp_pos(
            Eigen::internal::random(-2.0, trajectory_length_ + 2.0),
            Eigen::internal::random(-2.0, 2.0),
            Eigen::internal::random(landmark_distance_ - 0.5,
                                    landmark_distance_ + 0.5));
      landmarks_.push_back(tmp_pos);
      landmark_ids_.push_back(i);

      // Create the observations
      for (size_t k = 0; k < num_cams_; ++k) {
        Frame tmp_frame = frames_[k];
        // Randomly drop observations
        double rand_number = Eigen::internal::random(0.0, 1.0);
        if (rand_number > 0.95) {
          continue;
        }

        Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
        T_W_S.block<3,3>(0, 0) = tmp_frame.q_W_S.toRotationMatrix();
        T_W_S.block<3,1>(0, 3) = tmp_frame.p_W_S;
        Eigen::Matrix4d T_S_C = Eigen::Matrix4d::Identity();
        T_S_C.block<3,3>(0, 0) = tmp_frame.q_S_C.toRotationMatrix();
        T_S_C.block<3,1>(0, 3) = tmp_frame.p_S_C;
        Eigen::Matrix4d T_C_W = T_S_C.inverse() * T_W_S.inverse();
        Eigen::Vector3d l_C = T_C_W.block<3,3>(0, 0) * tmp_pos +
            T_C_W.block<3,1>(0, 3);
        Eigen::Vector2d proj;
        aslam::ProjectionResult result = camera_->project3(l_C, &proj);
        const bool projection_failed =
            (result == aslam::ProjectionResult::POINT_BEHIND_CAMERA) ||
            (result == aslam::ProjectionResult::PROJECTION_INVALID);
        if (projection_failed) {
          continue;
        }
        frames_[k].ids.push_back(i);
        frames_[k].observations.push_back(proj);
      }
    }
  }

  ceres::Problem problem_;

  std::shared_ptr<CameraType> camera_;

  Eigen::Vector3d zero_position_;
  Eigen::Quaterniond unit_quaternion_;

  Eigen::VectorXd distortion_params_;
  double fu_, fv_;
  double cu_, cv_;
  double res_u_, res_v_;
  double pixel_sigma_;

  // Store the simulated problem
  size_t num_cams_;
  size_t num_landmarks_;
  double landmark_distance_;
  double trajectory_length_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<
      Eigen::Vector3d>> landmarks_;
  std::vector<size_t> landmark_ids_;
  std::vector<Frame, Eigen::aligned_allocator<Frame>> frames_;


  // Ordering is [orientation position] -> [xyzw xyz].
  Eigen::Matrix<double, 7, 1> dummy_7d_0_;
  Eigen::Matrix<double, 7, 1> dummy_7d_1_;
};


TEST_F(SmartProjectionErrorTerms, test) {
  // Create a new camera
  constructCamera();
  constructProblem();

  // Initialize the parameters
  double** parameters = new double*[2* frames_.size()];
  double** jacobians = new double*[2 * frames_.size()];
  ceres::LocalParameterization* local_pose_parameterization =
      new robopt::local_param::PoseQuaternionLocalParameterization();
  for (size_t i = 0; i < frames_.size(); ++i) {
    parameters[2 * i] = new double [defs::pose::kPoseBlockSize];
    parameters[2 * i + 1] = new double [defs::pose::kPoseBlockSize];
    jacobians[2 * i] = new double [(2 * num_cams_ -3)  *
        defs::visual::kPoseBlockSize];
    jacobians[2 * i + 1] = new double [(2 * num_cams_ -3) *
        defs::visual::kPoseBlockSize];
    parameters[2 * i] = new double [defs::pose::kPoseBlockSize];
    parameters[2 * i + 1] = new double [defs::pose::kPoseBlockSize];
    jacobians[2 * i] = new double [(2 * num_cams_ -3)  *
        defs::visual::kPoseBlockSize];
    jacobians[2 * i + 1] = new double [(2 * num_cams_ -3) *
        defs::visual::kPoseBlockSize];

    Eigen::Map<Eigen::Vector3d> p_W_S(
          parameters[2 * i] + defs::pose::kOrientationBlockSize);
    p_W_S = frames_[i].p_W_S_init;
    Eigen::Map<Eigen::Quaterniond> q_W_S(parameters[2 * i]);
    q_W_S = frames_[i].q_W_S_init;
    Eigen::Map<Eigen::Vector3d> p_S_C(
          parameters[2 * i + 1] + defs::pose::kOrientationBlockSize);
    p_S_C = frames_[i].p_S_C_init;
    Eigen::Map<Eigen::Quaterniond> q_S_C(parameters[2 * i + 1]);
    q_S_C = frames_[i].q_S_C_init;

    problem_.AddParameterBlock(parameters[2 * i], defs::pose::kPoseBlockSize,
        local_pose_parameterization);
    if (i <= 1) {
      problem_.SetParameterBlockConstant(parameters[2 * i]);
    }
    problem_.AddParameterBlock(parameters[2 * i + 1],
        defs::pose::kPoseBlockSize, local_pose_parameterization);
    problem_.SetParameterBlockConstant(parameters[2 * i + 1]);
  }


  // Create a sample factor
  for (const size_t& id : landmark_ids_) {
    // Find the frames that observe this landmark
    std::vector<Frame, Eigen::aligned_allocator<Frame>> matching_frames;
    matching_frames.reserve(num_cams_);
    std::vector<CameraType*> matching_cameras;
    matching_cameras.reserve(num_cams_);
    for (const Frame& frame_i : frames_) {
      auto find_itr = std::find(frame_i.ids.begin(), frame_i.ids.end(), id);
      if (find_itr != frame_i.ids.end()) {
        matching_frames.push_back(frame_i);
        matching_cameras.push_back(camera_.get());
      }
    }

    // Initialize the parameters
    const size_t num_obs_i = matching_frames.size();
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        measurements_i;
    std::vector<double*> param_vec;
    for (size_t i = 0; i < num_obs_i; ++i) {
      // Push the parameters to the vector
      param_vec.push_back(parameters[2 * matching_frames[i].frame_id]);
      param_vec.push_back(parameters[2 * matching_frames[i].frame_id + 1]);

      for (size_t k = 0; k < matching_frames[i].ids.size(); ++k) {
        const size_t obs_id = matching_frames[i].ids[k];
        if (obs_id == id) {
          measurements_i.push_back(matching_frames[i].observations[k]);
          break;
        }
      }
    }

    // Now we are ready to create a factor
    ErrorTerm* error_term =
        new ErrorTerm(measurements_i, 2.0, matching_cameras);
//    error_term->setInitialGuess(landmarks_[id] +
//            Eigen::Vector3d(Eigen::internal::random(-0.5, 0.5),
//                            Eigen::internal::random(-0.5, 0.5),
//                            Eigen::internal::random(-0.5,0.5)));

    problem_.AddResidualBlock(error_term, NULL, param_vec);
  }

  // Solve the problem
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 50;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_, &summary);
  std::cout << summary.FullReport() << std::endl;

  // Check the result
  for (size_t i = 0; i < frames_.size(); ++i) {
    Eigen::Map<Eigen::Vector3d> p_W_S(
          parameters[2 * i] + defs::pose::kOrientationBlockSize);
    Eigen::Map<Eigen::Quaterniond> q_W_S(parameters[2 * i]);
    Eigen::Map<Eigen::Vector3d> p_S_C(
          parameters[2 * i + 1] + defs::pose::kOrientationBlockSize);
    Eigen::Map<Eigen::Quaterniond> q_S_C(parameters[2 * i + 1]);

    Eigen::Matrix4d T_W_S_gt = Eigen::Matrix4d::Identity();
    T_W_S_gt.block<3,3>(0, 0) = frames_[i].q_W_S.toRotationMatrix();
    T_W_S_gt.block<3,1>(0, 3) = frames_[i].p_W_S;
    Eigen::Matrix4d T_S_C_gt = Eigen::Matrix4d::Identity();
    T_S_C_gt.block<3,3>(0, 0) = frames_[i].q_S_C.toRotationMatrix();
    T_S_C_gt.block<3,1>(0, 3) = frames_[i].p_S_C;

    Eigen::Matrix4d T_W_S_res = Eigen::Matrix4d::Identity();
    T_W_S_res.block<3,3>(0, 0) = q_W_S.toRotationMatrix();
    T_W_S_res.block<3,1>(0, 3) = p_W_S;
    Eigen::Matrix4d T_S_C_res = Eigen::Matrix4d::Identity();
    T_S_C_res.block<3,3>(0, 0) = q_S_C.toRotationMatrix();
    T_S_C_res.block<3,1>(0, 3) = p_S_C;

    CHECK_EIGEN_MATRIX_NEAR(T_W_S_res, T_W_S_gt, 1e-6);
    CHECK_EIGEN_MATRIX_NEAR(T_S_C_res, T_S_C_gt, 1e-6);
  }
}

}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
