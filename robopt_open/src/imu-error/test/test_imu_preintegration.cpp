#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <common/typedefs.h>
#include <local-parameterization/pose-quaternion-local-param.h>
#include <imu-error/preintegration-factor.h>

using robopt::imu::PreintegrationBase;
using robopt::imu::PreintegrationFactor;

namespace robopt {

class PreintegrationErrorTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PreintegrationErrorTerms() {
    srand((unsigned int) time(0));
  }

protected:
  typedef PreintegrationBase Base;
  typedef PreintegrationFactor ErrorTerm;

  virtual void SetUp() {
    const size_t numFrames = 20; // meaning 1s!
    const size_t numMPs = 1000;
    const double dt = 1.0/imuParameters.rate;
    const size_t numIMU = numFrames*imuParameters.rate/10; // 20Hz frequency
    vPreintBase_.reserve(numFrames - 1);

    // generate random motion
    const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circ. freq.
    const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circ. freq.
    const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circ. freq.
    const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
    const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
    const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
    const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
    const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
    const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
    const double w_a_W_x = Eigen::internal::random(0.1,10.0);
    const double w_a_W_y = Eigen::internal::random(0.1,10.0);
    const double w_a_W_z = Eigen::internal::random(0.1,10.0);
    const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
    const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
    const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
    const double m_a_W_x = Eigen::internal::random(0.1,4.0);
    const double m_a_W_y = Eigen::internal::random(0.1,4.0);
    const double m_a_W_z = Eigen::internal::random(0.1,4.0);

    // states
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    Eigen::Quaterniond q(Eigen::AngleAxisd(axis.norm(), axis.normalized()));
    Eigen::Matrix4d Tws = Eigen::Matrix4d::Identity();
    Tws.block<3,3>(0,0) = q.toRotationMatrix();
    Tws.block<3,1>(0,3) = Eigen::Vector3d::Random();
    Eigen::Vector3d r = Tws.block<3,1>(0,3);
    Eigen::VectorXd speedAndBias(9);
    imuParameters.biasA = Eigen::Vector3d::Random() * imuParameters.sigma_ba;
    imuParameters.biasG = Eigen::Vector3d::Random() * imuParameters.sigma_bg;
    speedAndBias.setZero();
    Eigen::Vector3d v = speedAndBias.head<3>() = Eigen::Vector3d::Random();
    speedAndBias.segment<3>(3) = imuParameters.biasA;
    speedAndBias.tail<3>() = imuParameters.biasG;

    Matrix44Vector Tws_gt;
    const Eigen::Matrix4d Tws0 = Tws;
    Vector3Vector accMeasTmp;
    Vector3Vector gyrMeasTmp;
    std::vector<Vector3Vector> accMeas;
    std::vector<Vector3Vector> gyrMeas;
    Eigen::Matrix4d Tws_end;
    for (size_t i = 0; i < numIMU; ++i) {
      double time = double(i)/imuParameters.rate;
      if ((i%10) == 0) {
        Tws_gt.push_back(Tws);
        vTwsGT_.push_back(Tws);
        accMeas.push_back(accMeasTmp);
        gyrMeas.push_back(gyrMeasTmp);
        if (accMeasTmp.size() > 0) {
          PreintegrationBase* tmpPreintBase = new PreintegrationBase(
              accMeasTmp[0], gyrMeasTmp[0], imuParameters.biasA + Eigen::Vector3d::Random()*0.02,
              imuParameters.biasG + Eigen::Vector3d::Random()*0.02, imuParameters.sigma_a_c,
              imuParameters.sigma_g_c, imuParameters.sigma_aw_c,
              imuParameters.sigma_gw_c, imuParameters.g);
          for (size_t k = 0; k < accMeasTmp.size(); ++k) {
            tmpPreintBase->push_back(dt, accMeasTmp[k], gyrMeasTmp[k]);
          }
          vPreintBase_.push_back(tmpPreintBase);
        }
        accMeasTmp.clear();
        gyrMeasTmp.clear();
      }

      Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
          m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
          m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
      Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
          m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
          m_a_W_z*sin(w_a_W_z*time+p_a_W_z));
      Eigen::Quaterniond dq;

      // propagate orientation
      const double theta_half = omega_S.norm()*dt*0.5;
      const double sinc_theta_half = sinc_test(theta_half);
      const double cos_theta_half = std::cos(theta_half);
      dq.vec()=sinc_theta_half*0.5*dt*omega_S;
      dq.w()=cos_theta_half;
      q = q * dq;

      // propagate speed
      v+=dt*a_W;

      // propagate position
      r+=dt*v;

      Tws.block<3,3>(0,0) = q.toRotationMatrix();
      Tws.block<3,1>(0,3) = r;

      if (i + 1 == 10) {
        Tws_end = Tws;
      }
      speedAndBias.head<3>() = v;
      speedAndBias.block<3,1>(3,0) = imuParameters.biasA;
      speedAndBias.tail<3>() = imuParameters.biasG;
      vSpeedBiasGT_.push_back(speedAndBias);

      // generate measurements
      Eigen::Vector3d gyr = omega_S + imuParameters.sigma_g_c/sqrt(dt) *
          Eigen::Vector3d::Random() + imuParameters.biasG;
      Eigen::Vector3d acc = Tws.block<3,3>(0,0).transpose() *
          (a_W + Eigen::Vector3d(0,0,imuParameters.g)) +
          imuParameters.sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random() +
          imuParameters.biasA;
      accMeasTmp.push_back(acc);
      gyrMeasTmp.push_back(gyr);
    }
  }

  void GenerateRandomPose(
      double scale, Eigen::Quaterniond* q, Eigen::Vector3d* t) {
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    Eigen::Quaterniond q_tmp(Eigen::AngleAxisd(axis.norm(), axis.normalized()));
    (*q) = q_tmp;
    (*t) = Eigen::Vector3d::Random() * scale;
  }

  double sinc_test(double x) {
    if(std::fabs(x)>1e-10) {
      return std::sin(x)/x;
    } else {
      static const double c_2 = 1.0/6.0;
      static const double c_4 = 1.0/120.0;
      static const double c_6 = 1.0/5040.0;
      const double x_2 = x * x;
      const double x_4 = x_2 * x_2;
      const double x_6 = x_2 * x_2 * x_2;
      return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
    }
  }

  ceres::Problem problem_;
  ceres::Solver::Summary summary_;
  ceres::Solver::Options options_;

  Matrix44Vector vTwsGT_;
  Vector9Vector vSpeedBiasGT_;
  std::vector<PreintegrationBase*> vPreintBase_;


  struct ImuParameters{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double a_max = 1000.0;  ///< Accelerometer saturation. [m/s^2]
    double g_max = 1000.0;  ///< Gyroscope saturation. [rad/s]
    double sigma_g_c = 6.0e-3;  ///< Gyroscope noise density.
    double sigma_bg  = 0.05;  ///< Initial gyroscope bias.
    Eigen::Vector3d biasA;
    Eigen::Vector3d biasG;
    double sigma_a_c = 2.0e-2;  ///< Accelerometer noise density.
    double sigma_ba = 0.1;  ///< Initial accelerometer bias
    double sigma_gw_c = 3.0e-4; ///< Gyroscope drift noise density.
    double sigma_aw_c = 2.0e-3; ///< Accelerometer drift noise density.
    double g = 9.81;  ///< Earth acceleration.
    Eigen::Vector3d a0 = Eigen::Vector3d::Zero();
      ///< Mean of the prior accelerometer bias.
    int rate = 200;  ///< IMU rate in Hz.

  } imuParameters;

};


TEST_F(PreintegrationErrorTerms, Jacobians) {
  for (size_t i = 0; i < vPreintBase_.size(); ++i) {
    // Extract the preintegration base
    Base* tmpBase = vPreintBase_[i];

    // Create the raw pointers to the data
    double** parameters_nom = new double*[4];
    parameters_nom[0] = new double [defs::pose::kPoseBlockSize];
    parameters_nom[1] = new double [defs::pose::kSpeedBiasBlockSize];
    parameters_nom[2] = new double [defs::pose::kPoseBlockSize];
    parameters_nom[3] = new double [defs::pose::kSpeedBiasBlockSize];
    double** parameters_dist = new double*[4];
    parameters_dist[0] = new double [defs::pose::kPoseBlockSize];
    parameters_dist[1] = new double [defs::pose::kSpeedBiasBlockSize];
    parameters_dist[2] = new double [defs::pose::kPoseBlockSize];
    parameters_dist[3] = new double [defs::pose::kSpeedBiasBlockSize];

    // Create the poses and imu states for the nominal state
    const Eigen::Matrix4d T_W_S1 = vTwsGT_[i];
    const Eigen::Matrix4d T_W_S2 = vTwsGT_[i + 1];
    Eigen::Map<Eigen::Vector3d> p_W_S1(
          parameters_nom[0] + defs::pose::kOrientationBlockSize);
    p_W_S1 = T_W_S1.block<3,1>(0,3);
    Eigen::Map<Eigen::Quaterniond> q_W_S1(parameters_nom[0]);
    q_W_S1 = Eigen::Quaterniond(T_W_S1.block<3,3>(0,0));
    Eigen::Map<Vector9d> speed_bias1(parameters_nom[1]);
    speed_bias1 = vSpeedBiasGT_[i];
    Eigen::Map<Eigen::Vector3d> p_W_S2(
          parameters_nom[2] + defs::pose::kOrientationBlockSize);
    p_W_S2 = T_W_S2.block<3,1>(0,3);
    Eigen::Map<Eigen::Quaterniond> q_W_S2(parameters_nom[2]);
    q_W_S2 = Eigen::Quaterniond(T_W_S2.block<3,3>(0,0));
    Eigen::Map<Vector9d> speed_bias2(parameters_nom[3]);
    speed_bias2 = vSpeedBiasGT_[i + 1];

    // Create the analytical jacobians
    double** jacobians_analytical = new double *[4];
    jacobians_analytical[0] = new double [defs::pose::kPoseBlockSize * 15];
    jacobians_analytical[1] = new double [defs::pose::kSpeedBiasBlockSize * 15];
    jacobians_analytical[2] = new double [defs::pose::kPoseBlockSize * 15];
    jacobians_analytical[3] = new double [defs::pose::kSpeedBiasBlockSize * 15];
    Eigen::Map<Eigen::Matrix<double, 15,
              defs::pose::kPoseBlockSize, Eigen::RowMajor>>
        J_res_wrt_T_W_S1(jacobians_analytical[0]);
    Eigen::Map<Eigen::Matrix<double, 15, defs::pose::kSpeedBiasBlockSize,
        Eigen::RowMajor>> J_res_wrt_speed_bias1(jacobians_analytical[1]);
    Eigen::Map<Eigen::Matrix<double, 15,
              defs::pose::kPoseBlockSize, Eigen::RowMajor>>
        J_res_wrt_T_W_S2(jacobians_analytical[2]);
    Eigen::Map<Eigen::Matrix<double, 15, defs::pose::kSpeedBiasBlockSize,
        Eigen::RowMajor>> J_res_wrt_speed_bias2(jacobians_analytical[3]);

    ErrorTerm error_term(tmpBase);
    double* residuals_nom = new double[15];
    error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual_map_nom(residuals_nom);

    // Map the distorted parameters
    Eigen::Map<Eigen::Quaterniond> q_W_S1_dist(parameters_dist[0]);
    q_W_S1_dist = q_W_S1;
    Eigen::Map<Eigen::Vector3d> p_W_S1_dist(
          parameters_dist[0] + defs::pose::kOrientationBlockSize);
    p_W_S1_dist = p_W_S1;
    Eigen::Map<Vector9d> speed_bias1_dist(parameters_dist[1]);
    speed_bias1_dist = speed_bias1;
    Eigen::Map<Eigen::Quaterniond> q_W_S2_dist(parameters_dist[2]);
    q_W_S2_dist = q_W_S2;
    Eigen::Map<Eigen::Vector3d> p_W_S2_dist(
          parameters_dist[2] + defs::pose::kOrientationBlockSize);
    p_W_S2_dist = p_W_S2;
    Eigen::Map<Vector9d> speed_bias2_dist(parameters_dist[3]);
    speed_bias2_dist = speed_bias2;

    // Create the numerical jacobians
    const double delta = 1e-7;
    double* residuals_dist = new double[15];
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual_map_dist(residuals_dist);
    Eigen::Matrix<double, 15, defs::pose::kPoseBlockSize> J_res_wrt_T_W_S1_num;
    Eigen::Matrix<double, 15, defs::pose::kSpeedBiasBlockSize>
        J_res_wrt_speed_bias1_num;
    Eigen::Matrix<double, 15, defs::pose::kPoseBlockSize> J_res_wrt_T_W_S2_num;
    Eigen::Matrix<double, 15, defs::pose::kSpeedBiasBlockSize>
        J_res_wrt_speed_bias2_num;
    Eigen::Matrix<double, 15, 1>  difference;

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
      speed_bias1_dist = speed_bias1;
      speed_bias2_dist = speed_bias2;

      error_term.Evaluate(parameters_dist, residuals_dist, NULL);

      difference = residual_map_dist - residual_map_nom;
      J_res_wrt_T_W_S1_num.block<15, 1>(0, i) =
          difference / delta;
    }

    CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S1, J_res_wrt_T_W_S1_num,
                            J_res_wrt_T_W_S1_num.norm() *1e-5);

    // Jacobian for speed_bias1
    for (size_t i = 0; i < 9; ++i) {
      Eigen::VectorXd dist(9);
      dist.setZero();
      dist[i] = delta;

      // Compute the disturbed parameters
      q_W_S1_dist = q_W_S1;
      p_W_S1_dist = p_W_S1;
      q_W_S2_dist = q_W_S2;
      p_W_S2_dist = p_W_S2;
      speed_bias1_dist = speed_bias1 + dist;
      speed_bias2_dist = speed_bias2;

      error_term.Evaluate(parameters_dist, residuals_dist, NULL);

      difference = residual_map_dist - residual_map_nom;
      J_res_wrt_speed_bias1_num.block<15, 1>(0, i) =
          difference / delta;
    }

    // TODO: shouldn't tolerance be lower?
    CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_speed_bias1, J_res_wrt_speed_bias1_num,
                            J_res_wrt_speed_bias1_num.norm() *1e-5);

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
      speed_bias1_dist = speed_bias1;
      speed_bias2_dist = speed_bias2;

      error_term.Evaluate(parameters_dist, residuals_dist, NULL);

      difference = residual_map_dist - residual_map_nom;
      J_res_wrt_T_W_S2_num.block<15, 1>(0, i) =
          difference / delta;
    }

    CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_T_W_S2, J_res_wrt_T_W_S2_num,
                             J_res_wrt_T_W_S2_num.norm() *1e-5);

    // Jacobian for speed_bias1
    for (size_t i = 0; i < 9; ++i) {
      Eigen::VectorXd dist(9);
      dist.setZero();
      dist[i] = delta;

      // Compute the disturbed parameters
      q_W_S1_dist = q_W_S1;
      p_W_S1_dist = p_W_S1;
      q_W_S2_dist = q_W_S2;
      p_W_S2_dist = p_W_S2;
      speed_bias1_dist = speed_bias1;
      speed_bias2_dist = speed_bias2 + dist;

      error_term.Evaluate(parameters_dist, residuals_dist, NULL);

      difference = residual_map_dist - residual_map_nom;
      J_res_wrt_speed_bias2_num.block<15, 1>(0, i) =
          difference / delta;
    }

    CHECK_EIGEN_MATRIX_NEAR(J_res_wrt_speed_bias2, J_res_wrt_speed_bias2_num,
                            J_res_wrt_speed_bias2_num.norm() *1e-5);
  }
}


} // namespace robopt


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
