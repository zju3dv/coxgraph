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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * measurements.hpp
 * @brief Storage containers for measurements.
 * @author: Marco Karrer
 * Created on: Aug 15, 2018
 */

#pragma once

#include "typedefs.hpp"

/// \brief pgbe The main namespace of this package
namespace pgbe {

struct OdomMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OdomMeasurement()
      : timestamp(0.0),
        translation(Eigen::Vector3d::Zero()),
        rotation(Eigen::Quaterniond::Identity()) {}
  OdomMeasurement(const double timestamp_, const Eigen::Vector3d& translation_,
                  const Eigen::Quaterniond& rotation_)
      : timestamp(timestamp_), translation(translation_), rotation(rotation_) {}
  double timestamp;
  Eigen::Vector3d translation;  // Imu-to-World
  Eigen::Quaterniond rotation;  // Imu-to-World
};

typedef std::deque<OdomMeasurement, Eigen::aligned_allocator<OdomMeasurement>>
    OdomMeasurementQueue;

struct GPSmeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GPSmeasurement()
      : timestamp(0.0),
        raw_measurement(Eigen::Vector3d::Zero()),
        local_measurement(Eigen::Vector3d::Zero()),
        converted(false) {}
  GPSmeasurement(const double timestamp_,
                 const Eigen::Vector3d& raw_measurement_,
                 const Eigen::Vector3d& local_measurement_)
      : timestamp(timestamp_),
        raw_measurement(raw_measurement_),
        local_measurement(local_measurement_),
        converted(true) {}
  double timestamp;
  Eigen::Vector3d raw_measurement;    //[latitude, longitude, altitude]
  Eigen::Vector3d local_measurement;  //[east, north, up]
  bool converted;
  Eigen::Matrix3d covariance;
};

typedef std::deque<GPSmeasurement, Eigen::aligned_allocator<GPSmeasurement>>
    GPSmeasurementQueue;

struct OdomGPScombined {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OdomGPScombined()
      : timestamp(0.0), odometry(OdomMeasurement()), gps(GPSmeasurement()) {}
  OdomGPScombined(const OdomMeasurement& odometry_, const GPSmeasurement& gps_)
      : timestamp(odometry_.timestamp), odometry(odometry_), gps(gps_) {}
  double timestamp;
  OdomMeasurement odometry;
  GPSmeasurement gps;
};

typedef std::deque<OdomGPScombined, Eigen::aligned_allocator<OdomGPScombined>>
    OdomGPScombinedQueue;

typedef std::vector<OdomGPScombined, Eigen::aligned_allocator<OdomGPScombined>>
    OdomGPScombinedVector;

struct PclMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp;
  Vector3Vector points;
  Vector3Vector colors;
};

typedef std::deque<PclMeasurement, Eigen::aligned_allocator<PclMeasurement>>
    PclMeasurementQueue;

struct OdomPclCombined {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp;
  PclMeasurement pcl;
  OdomMeasurement odometry;
};

typedef std::deque<OdomPclCombined, Eigen::aligned_allocator<OdomPclCombined>>
    OdomPclCombinedQueue;

}  // namespace pgbe
