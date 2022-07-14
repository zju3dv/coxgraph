#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <eigen-checks/glog.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <common/definitions.h>
#include <common/common.h>
#include <common/typedefs.h>
#include <local-parameterization/pose-quaternion-local-param.h>
#include <imu-error/preintegration-base.h>

using robopt::imu::PreintegrationBase;

namespace robopt {

namespace imu {

class PreintegrationBaseTerms : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PreintegrationBaseTerms() {
    srand((unsigned int) time(0));
  }

protected:
  typedef PreintegrationBase Base;

  virtual void SetUp() {
    const int numFrames = 10;
    const double dt = 1.0/imuParameters.rate;
    const int numIMU = numFrames*imuParameters.rate/2; // 1Hz frequency
    const int measPerFrame = numIMU/numFrames;

    vPreintBase_.reserve(numFrames - 1);
    vAccelerationMeasured_.reserve(numFrames - 1);
    vAccelerationGT_.reserve(numFrames - 1);
    vGyroscopeMeasured_.reserve(numFrames - 1);
    vGyroscopeGT_.reserve(numFrames - 1);

    // Initialize the ground truth biases
    imuParameters.biasA = Eigen::Vector3d::Random() * 0.3;
    imuParameters.biasG = Eigen::Vector3d::Random() * 0.2;

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
    speedAndBias.setZero();
    Eigen::Vector3d v = speedAndBias.head<3>() = Eigen::Vector3d::Random();
    speedAndBias.block<3,1>(3,0) = imuParameters.biasA;
    speedAndBias.tail<3>() = imuParameters.biasG;
    vSpeedBiasGT_.push_back(speedAndBias);
    vTwsGT_.push_back(Tws);

    Matrix44Vector Tws_gt;
    const Eigen::Matrix4d Tws0 = Tws;
    Vector3Vector accMeasTmp;
    Vector3Vector accGtTmp;
    Vector3Vector gyrMeasTmp;
    Vector3Vector gyrGtTmp;
    Eigen::Matrix4d Tws_end;
    for (size_t i = 0; i < numIMU; ++i) {
      double time = double(i)/imuParameters.rate;
      if ((i%measPerFrame) == 0) {
        if (accGtTmp.size() > 0) {
          Tws_gt.push_back(Tws);
          vTwsGT_.push_back(Tws);
          vSpeedBiasGT_.push_back(speedAndBias);
          vAccelerationGT_.push_back(accGtTmp);
          vAccelerationMeasured_.push_back(accMeasTmp);
          vGyroscopeGT_.push_back(gyrGtTmp);
          vGyroscopeMeasured_.push_back(gyrMeasTmp);
        }
        accGtTmp.clear();
        accMeasTmp.clear();
        gyrGtTmp.clear();
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

      if (i + 1 == measPerFrame) {
        Tws_end = Tws;
      }
      speedAndBias.head<3>() = v;
      speedAndBias.block<3,1>(3,0) = imuParameters.biasA;
      speedAndBias.tail<3>() = imuParameters.biasG;

      // generate measurements
      Eigen::Vector3d accGT = Tws.block<3,3>(0,0).transpose() *
          (a_W + Eigen::Vector3d(0,0,imuParameters.g));
      Eigen::Vector3d acc = accGT +
          imuParameters.sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random() +
          imuParameters.biasA;
      Eigen::Vector3d gyr = omega_S + imuParameters.sigma_g_c/sqrt(dt) *
          Eigen::Vector3d::Random() + imuParameters.biasG;

      accGtTmp.push_back(accGT);
      accMeasTmp.push_back(acc);
      gyrGtTmp.push_back(omega_S);
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

  Matrix44Vector vTwsGT_;
  Vector9Vector vSpeedBiasGT_;
  std::vector<Vector3Vector> vAccelerationMeasured_;
  std::vector<Vector3Vector> vAccelerationGT_;
  std::vector<Vector3Vector> vGyroscopeMeasured_;
  std::vector<Vector3Vector> vGyroscopeGT_;
  std::vector<PreintegrationBase*> vPreintBase_;


  struct ImuParameters{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double a_max = 1000.0;  ///< Accelerometer saturation. [m/s^2]
    double g_max = 1000.0;  ///< Gyroscope saturation. [rad/s]
    double sigma_g_c = 6.0e-4;  ///< Gyroscope noise density.
    double sigma_bg  = 0.05;  ///< Initial gyroscope bias.
    Eigen::Vector3d biasA;
    Eigen::Vector3d biasG;
    double sigma_a_c = 2.0e-3;  ///< Accelerometer noise density.
    double sigma_ba = 0.1;  ///< Initial accelerometer bias
    double sigma_gw_c = 3.0e-6; ///< Gyroscope drift noise density.
    double sigma_aw_c = 2.0e-5; ///< Accelerometer drift noise density.
    double g = 9.81;  ///< Earth acceleration.
    Eigen::Vector3d a0 = Eigen::Vector3d::Zero();
      ///< Mean of the prior accelerometer bias.
    int rate = 200;  ///< IMU rate in Hz.

  } imuParameters;


};



TEST_F(PreintegrationBaseTerms, NoiseFreeIntegration) {
  // Creat the preintegration bases
  const size_t numFrames = vAccelerationGT_.size();

  for (size_t i = 0; i < numFrames; ++i) {
    Vector3Vector acc = vAccelerationGT_[i];
    Vector3Vector gyr = vGyroscopeGT_[i];
    Base* tmpBase = new Base(acc[0], gyr[0], Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), imuParameters.sigma_a_c,
        imuParameters.sigma_g_c, imuParameters.sigma_aw_c,
        imuParameters.sigma_gw_c, imuParameters.g);
    for (size_t k = 0; k < acc.size(); ++k) {
      tmpBase->push_back(1.0/imuParameters.rate, acc[k], gyr[k]);
    }

    // Get the ground truth relative Pose
    Eigen::Matrix4d Tws1 = vTwsGT_[i];
    Eigen::Matrix4d Tws2 = vTwsGT_[i + 1];
    Eigen::Matrix4d deltaT = Tws1.inverse()*Tws2;
    Eigen::Quaterniond delta_q_gt(deltaT.block<3,3>(0,0));
    Eigen::Vector4d delta_q_gt_v = delta_q_gt.coeffs();
    Eigen::Vector3d delta_p_gt = deltaT.block<3,1>(0,3);

    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;
    Eigen::Quaterniond q_W_S1(Tws1.block<3,3>(0,0));
    tmpBase->getDelta(Tws1.block<3,1>(0,3), q_W_S1,
          vSpeedBiasGT_[i].head<3>(), Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(), &delta_p, &delta_q, &delta_v);
    Eigen::Vector4d delta_q_v = delta_q.coeffs();
    Eigen::Vector3d delta_v_gt = q_W_S1.inverse() *
        (vSpeedBiasGT_[i+1].head<3>() - vSpeedBiasGT_[i].head<3>());

    delete tmpBase;
    CHECK_EIGEN_MATRIX_NEAR(delta_p_gt, delta_p, 1e-2);
    CHECK_EIGEN_MATRIX_NEAR(delta_q_gt_v, delta_q_v, 1e-2);
    CHECK_EIGEN_MATRIX_NEAR(delta_v_gt, delta_v, 5e-2);
  }
}

TEST_F(PreintegrationBaseTerms, IntegrationJacobianState) {
  Vector3Vector acc = vAccelerationMeasured_[0];
  Vector3Vector gyr = vGyroscopeMeasured_[0];
  const double dt = 1.0/imuParameters.rate;
  Base* tmpBase1 = new Base(acc[0], gyr[0], imuParameters.biasA,
      imuParameters.biasG, imuParameters.sigma_a_c,
      imuParameters.sigma_g_c, imuParameters.sigma_aw_c,
      imuParameters.sigma_gw_c, imuParameters.g);
  for (size_t k = 0; k < 5; ++k) {
    tmpBase1->push_back(dt, acc[k], gyr[k]);
  }
  Eigen::Vector3d result_p_nom;
  Eigen::Quaterniond result_q_nom;
  Eigen::Vector3d result_v_nom;
  Eigen::Vector3d result_b_a_nom;
  Eigen::Vector3d result_b_g_nom;
  tmpBase1->midPointIntegration(dt, &result_p_nom, &result_q_nom,
      &result_v_nom, true);
  result_b_a_nom = tmpBase1->linear_bias_a_;
  result_b_g_nom = tmpBase1->linear_bias_g_;

  const double delta = 1e-6;
  Eigen::Matrix<double, 15, 15> J_num;
  J_num.setZero();
  for (size_t i = 0; i < 15; ++i) {
    Eigen::VectorXd dist(15);
    dist.setZero();
    dist[i] = delta;
    Base* tmpBase2 = new Base(acc[0], gyr[0], imuParameters.biasA,
        imuParameters.biasG, imuParameters.sigma_a_c,
        imuParameters.sigma_g_c, imuParameters.sigma_aw_c,
        imuParameters.sigma_gw_c, imuParameters.g);
    for (size_t k = 0; k < 5; ++k) {
      tmpBase2->push_back(dt, acc[k], gyr[k]);
    }
    Eigen::Vector3d result_p_dist;
    Eigen::Quaterniond result_q_dist;
    Eigen::Vector3d result_v_dist;
    Eigen::Vector3d result_b_a_dist;
    Eigen::Vector3d result_b_g_dist;
    common::quaternion::Plus(tmpBase2->delta_q_,
        dist.segment<3>(defs::pose::StateOrder::kRotation),
        &tmpBase2->delta_q_);
    tmpBase2->delta_p_ += dist.segment<3>(defs::pose::StateOrder::kPosition);
    tmpBase2->delta_v_ += dist.segment<3>(defs::pose::StateOrder::kVelocity);
    tmpBase2->linear_bias_a_ += dist.segment<3>(defs::pose::StateOrder::kBiasA);
    tmpBase2->linear_bias_g_ += dist.segment<3>(defs::pose::StateOrder::kBiasG);

    tmpBase2->midPointIntegration(dt, &result_p_dist, &result_q_dist,
        &result_v_dist, true);
    result_b_a_dist = tmpBase2->linear_bias_a_;
    result_b_g_dist = tmpBase2->linear_bias_g_;

    Eigen::VectorXd difference(15);
    difference.setZero();
    Eigen::Vector3d tmpRotDiff;
    common::quaternion::Minus(result_q_dist, result_q_nom, &tmpRotDiff);
    difference.segment<3>(defs::pose::StateOrder::kRotation) = tmpRotDiff;
    difference.segment<3>(defs::pose::StateOrder::kPosition) =
        result_p_dist - result_p_nom;
    difference.segment<3>(defs::pose::StateOrder::kVelocity) =
        result_v_dist - result_v_nom;
    difference.segment<3>(defs::pose::StateOrder::kBiasA) =
        result_b_a_dist - result_b_a_nom;
    difference.segment<3>(defs::pose::StateOrder::kBiasG) =
        result_b_g_dist - result_b_g_nom;

    J_num.block<15,1>(0,i) = difference/delta;
  }

  CHECK_EIGEN_MATRIX_NEAR(tmpBase1->step_jacobian_, J_num, 1e-6);
}

TEST_F(PreintegrationBaseTerms, IntegrationJacobianNoise) {
  Vector3Vector acc = vAccelerationMeasured_[0];
  Vector3Vector gyr = vGyroscopeMeasured_[0];
  const double dt = 1.0/imuParameters.rate;
  Base* tmpBase1 = new Base(acc[0], gyr[0], imuParameters.biasA,
      imuParameters.biasG, imuParameters.sigma_a_c,
      imuParameters.sigma_g_c, imuParameters.sigma_aw_c,
      imuParameters.sigma_gw_c, imuParameters.g);
  for (size_t k = 0; k < 5; ++k) {
    tmpBase1->push_back(dt, acc[k], gyr[k]);
  }
  Eigen::Vector3d result_p_nom;
  Eigen::Quaterniond result_q_nom;
  Eigen::Vector3d result_v_nom;
  Eigen::Vector3d result_b_a_nom;
  Eigen::Vector3d result_b_g_nom;
  tmpBase1->midPointIntegration(dt, &result_p_nom, &result_q_nom,
      &result_v_nom, true);
  result_b_a_nom = tmpBase1->linear_bias_a_;
  result_b_g_nom = tmpBase1->linear_bias_g_;
  Eigen::MatrixXd J_an = Eigen::MatrixXd::Zero(15,18);
  J_an = tmpBase1->step_V_;

  const double delta = 1e-4;
  Eigen::Matrix<double, 15, 18> J_num;
  J_num.setZero();
  for (size_t i = 0; i < 18; ++i) {
    Eigen::VectorXd dist(18);
    dist.setZero();
    dist[i] = delta;
    Base* tmpBase2 = new Base(acc[0], gyr[0], imuParameters.biasA,
        imuParameters.biasG, imuParameters.sigma_a_c,
        imuParameters.sigma_g_c, imuParameters.sigma_aw_c,
        imuParameters.sigma_gw_c, imuParameters.g);
    for (size_t k = 0; k < 5; ++k) {
      tmpBase2->push_back(dt, acc[k], gyr[k]);
    }
    Eigen::Vector3d result_p_dist;
    Eigen::Quaterniond result_q_dist;
    Eigen::Vector3d result_v_dist;
    Eigen::Vector3d result_b_a_dist;
    Eigen::Vector3d result_b_g_dist;

    tmpBase2->acc_0_ += dist.segment<3>(0);
    tmpBase2->gyr_0_ += dist.segment<3>(3);
    tmpBase2->acc_1_ += dist.segment<3>(6);
    tmpBase2->gyr_0_ += dist.segment<3>(9);
    tmpBase2->linear_bias_a_ += dist.segment<3>(12) * dt;
    tmpBase2->linear_bias_g_ += dist.segment<3>(15) * dt;
    tmpBase2->midPointIntegration(dt, &result_p_dist, &result_q_dist,
        &result_v_dist, true);
    result_b_a_dist = tmpBase2->linear_bias_a_;
    result_b_g_dist = tmpBase2->linear_bias_g_;

    Eigen::VectorXd difference(15);
    difference.setZero();
    Eigen::Vector3d tmpRotDiff;
    common::quaternion::Minus(result_q_dist, result_q_nom, &tmpRotDiff);
    difference.segment<3>(defs::pose::StateOrder::kRotation) = tmpRotDiff;
    difference.segment<3>(defs::pose::StateOrder::kPosition) =
        result_p_dist - result_p_nom;
    difference.segment<3>(defs::pose::StateOrder::kVelocity) =
        result_v_dist - result_v_nom;
    difference.segment<3>(defs::pose::StateOrder::kBiasA) =
        result_b_a_dist - result_b_a_nom;
    difference.segment<3>(defs::pose::StateOrder::kBiasG) =
        result_b_g_dist - result_b_g_nom;

    J_num.block<15,1>(0,i) = difference/delta;
  }

  CHECK_EIGEN_MATRIX_NEAR(J_an, J_num, 1e-4);
}

} // namespace imu

} // namespace robopt


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
