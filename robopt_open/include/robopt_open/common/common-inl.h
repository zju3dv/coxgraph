/*
* Copyright (c) 2018, Vision for Robotics Lab
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the Vision for Robotics Lab, ETH Zurich nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

/*
 * common-inl.h
 * @brief Inline Implementation for commonly used operations.
 * @author: Marco Karrer
 * Created on: Mar 19, 2018
 */


/// \brief robopt The main namespace of this package.
namespace robopt {

namespace common {

namespace weighting {

inline double cauchyWeight(const double resid, const double c) {
  double inv_r = std::max(std::numeric_limits<double>::min(), 1.0/resid);
  double influence = (resid)/((resid/c) * (resid/c) + 1.0);

  return influence * inv_r;
}

}

namespace quaternion {

inline Eigen::Matrix3d Gamma(const Eigen::Vector3d& phi) {
  const double phi_squared_norm = phi.squaredNorm();

  if (phi_squared_norm < 1e-6) {
    Eigen::Matrix3d gamma;
    gamma.setIdentity();
    gamma -= 0.5 * common::skew(phi);
    return gamma;
  }
  const double phi_norm = sqrt(phi_squared_norm);
  const Eigen::Matrix3d phi_skew(common::skew(phi));

  Eigen::Matrix3d gamma;
  gamma.setIdentity();
  gamma -= ((1.0 - std::cos(phi_norm)) / phi_squared_norm) * phi_skew;
  const double phi_cubed = (phi_norm * phi_squared_norm);
  gamma += ((phi_norm - std::sin(phi_norm)) / phi_cubed) * phi_skew * phi_skew;
  return gamma;
}

inline Eigen::Quaterniond ExpMap(const Eigen::Vector3d& theta) {
  const double theta_squared_norm = theta.squaredNorm();

  if (theta_squared_norm < 1e-6) {
    Eigen::Quaterniond q(
        1, theta(0) * 0.5, theta(1) * 0.5, theta(2) * 0.5);
    q.normalize();
    return q;
  }

  const double theta_norm = std::sqrt(theta_squared_norm);
  const Eigen::Vector3d q_imag =
      std::sin(theta_norm * 0.5) * theta / theta_norm;
  Eigen::Quaterniond q(
      std::cos(theta_norm * 0.5), q_imag(0), q_imag(1), q_imag(2));
  return q;
}

inline Eigen::Vector3d LogMap(const Eigen::Quaterniond& q) {
  const Eigen::Block<const Eigen::Vector4d, 3, 1> q_imag = q.vec();
  const double q_imag_squared_norm = q_imag.squaredNorm();

  if (q_imag_squared_norm < 1e-6) {
    return 2 * std::copysign(1, q.w()) * q_imag;
  }

  const double q_imag_norm = std::sqrt(q_imag_squared_norm);
  Eigen::Vector3d q_log = 2*std::atan2(q_imag_norm, q.w()) * 
    q_imag / q_imag_norm;
  return q_log;
}

inline void Plus(
    const Eigen::Ref<const Eigen::Vector4d>& q,
    const Eigen::Ref<const Eigen::Vector3d>& delta,
    Eigen::Quaterniond* q_res) {
  CHECK_NOTNULL(q_res);
  const Eigen::Map<const Eigen::Quaterniond> p_mapped(q.data());
  *q_res = p_mapped * ExpMap(delta);
}

inline void Plus(
    const Eigen::Quaterniond& q,
    const Eigen::Ref<const Eigen::Vector3d>& delta,
    Eigen::Quaterniond* q_res) {
  CHECK_NOTNULL(q_res);
  *q_res = q * ExpMap(delta);
}


inline void Minus(
    const Eigen::Quaterniond& p, const Eigen::Quaterniond& q,
    Eigen::Vector3d* p_minus_q) {
  CHECK_NOTNULL(p_minus_q);
  //*p_minus_q = LogMap(p * q.inverse());
  *p_minus_q = LogMap(q.inverse() * p);
}

inline Eigen::Quaterniond DeltaQ(
    const Eigen::Vector3d& theta) {
  Eigen::Vector3d half_theta = theta/2.0;
  Eigen::Quaterniond dq(1.0, half_theta[0], half_theta[1], half_theta[2]);
  return dq;
}

inline Eigen::Matrix4d PlusMat(const Eigen::Vector4d& q) {
  Eigen::Matrix4d Q;
  Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
  Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
  Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

inline Eigen::Matrix4d PlusMat(const Eigen::Quaterniond& q) {
  return PlusMat(q.coeffs());
}

inline Eigen::Matrix4d OPlusMat(const Eigen::Vector4d & q) {
  Eigen::Matrix4d Q;
  Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
  Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
  Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

inline Eigen::Matrix4d OPlusMat(const Eigen::Quaterniond& q) {
  return OPlusMat(q.coeffs());
}

} // namespace quaternion

inline Eigen::Matrix3d skew(
    const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector[2], vector[1],
          vector[2], 0.0, -vector[0],
          -vector[1], vector[0], 0.0;
  return matrix;
}

namespace yaw {

inline Eigen::Quaterniond ExpMap(const double yaw) {
  const double half_yaw = yaw/2.0;
  Eigen::Quaterniond q_res(std::cos(half_yaw), 0.0, 0.0, std::sin(half_yaw));

  return q_res;
}

inline double LogMap(const Eigen::Quaterniond& q) {
  return std::atan2(2 * (q.w() * q.z() + q.x() * q.y()),
                    1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
}

inline Eigen::Matrix<double, 1, 4> LogMapJacobian(
    const Eigen::Quaterniond& q) {
  Eigen::Matrix<double, 1, 4> J;
  // Intermediate variables
  const double qx = q.x();
  const double qy = q.y();
  const double qy2 = qy * qy;
  const double qz = q.z();
  const double qz2 = qz * qz;
  const double qw = q.w();

  // Some more complex expressions
  const double a = 2.0 * qy2 + 2.0 * qz2 - 1.0;
  const double b = 2.0 * qw * qz + 2.0 * qx * qy;
  const double den = (a * a + b * b);

  // d/dqx
  J(0,0) = -(2.0 * qy  * a)/den;
  // d/dqy
  J(0,1) = -(2.0 * qx * a - 4.0 * qy * b)/den;
  // d/dqz
  J(0,2) = -(2.0 * qw * a - 4.0 * qz * b)/den;
  // d/dqw
  J(0,3) = -2.0 * qz * a/den;

  return J;
}

inline void Plus(
    const Eigen::Ref<const Eigen::Vector4d>& q, const double& delta,
    Eigen::Quaterniond* q_res) {
  CHECK_NOTNULL(q_res);
  const Eigen::Map<const Eigen::Quaterniond> p_mapped(q.data());
  *q_res = ExpMap(delta) * p_mapped;
}

inline void Plus(
    const Eigen::Quaterniond& q, const double& delta,
    Eigen::Quaterniond* q_res) {
  CHECK_NOTNULL(q_res);
  *q_res =  ExpMap(delta) * q;
}

inline void Minus(
    const Eigen::Quaterniond& p, const Eigen::Quaterniond& q,
    double* p_minus_q) {
  CHECK_NOTNULL(p_minus_q);
  *p_minus_q = LogMap(p) - LogMap(q);
}

inline double normalizeYaw(const double yaw) {
  if (yaw > M_PI) {
    return yaw - 2 * M_PI;
  } else if (yaw < -M_PI) {
    return yaw + 2 * M_PI;
  } else {
    return yaw;
  }
}

} // namespace yaw

} // namepsace common

} // namespace robopt
