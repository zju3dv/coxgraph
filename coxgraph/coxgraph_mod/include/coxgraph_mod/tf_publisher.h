#ifndef COXGRAPH_MOD_TF_PUBLISHER_H_
#define COXGRAPH_MOD_TF_PUBLISHER_H_

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "coxgraph_mod/common.h"

namespace coxgraph {
namespace mod {

class TfPublisher {
 public:
  typedef std::shared_ptr<TfPublisher> Ptr;

  TfPublisher(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : nh_(nh), nh_private_(nh_private), current_time_(ros::Time::now()) {
    nh_private_.param<std::string>("odom_frame", odom_frame_, "odom");
    nh_private_.param<std::string>("sensor_frame", sensor_frame_, "cam");
    odom_pub_ = nh_private_.advertise<nav_msgs::Odometry>("odometry", 10, true);
    float tf_pub_period_ms = 10.0;
    nh_private_.param<float>("tf_pub_period_ms", tf_pub_period_ms,
                             tf_pub_period_ms);
    tf_timer_ =
        nh_.createTimer(ros::Duration(tf_pub_period_ms / 1000),
                        &TfPublisher::PublishPositionAsTransformCallback, this);

    nh_private_.param<std::string>("imu_frame", imu_frame_, imu_frame_);
    if (imu_frame_.size()) {
      std::vector<float> T_I_S_vec;
      nh_private_.param<std::vector<float>>("T_I_S", T_I_S_vec, T_I_S_vec);
      tf::Matrix3x3 rotation;
      tf::Vector3 translation;
      tfFromStdVector(T_I_S_vec, &rotation, &translation);
      T_I_S_ = TransformFromTf(rotation, translation);
    }
    nh_private_.param<std::string>("imu_frame", imu_frame_, imu_frame_);
    if (imu_frame_.size()) {
      std::vector<float> T_I_S_vec;
      nh_private_.param<std::vector<float>>("T_I_S", T_I_S_vec, T_I_S_vec);
      tf::Matrix3x3 rotation;
      tf::Vector3 translation;
      tfFromStdVector(T_I_S_vec, &rotation, &translation);
      T_I_S_ = TransformFromTf(rotation, translation);
    }
  }
  ~TfPublisher() = default;

  void updatePose(Eigen::Matrix4d pose, double timestamp) {
    std::lock_guard<std::mutex> pose_update_lock(pose_update_mutex_);
    T_O_S_ = TransformFromEigen(pose);
    current_time_.fromSec(timestamp);
  }

  void updatePose(cv::Mat pose, double timestamp) {
    if (pose.empty()) return;
    std::lock_guard<std::mutex> pose_update_lock(pose_update_mutex_);
    T_O_S_ = TransformFromCvMat(pose);
    current_time_.fromSec(timestamp);
  }

  void PublishPositionAsTransformCallback(const ros::TimerEvent& event) {
    std::lock_guard<std::mutex> pose_update_lock(pose_update_mutex_);
    if (imu_frame_.empty()) {
      tf_broadcaster_.sendTransform(tf::StampedTransform(
          T_O_S_, current_time_, odom_frame_, sensor_frame_));
    } else {
      tf::Transform T_O_I = T_O_S_ * T_I_S_.inverse();
      tf_broadcaster_.sendTransform(
          tf::StampedTransform(T_O_I, current_time_, odom_frame_, imu_frame_));
      tf_broadcaster_.sendTransform(tf::StampedTransform(
          T_I_S_, current_time_, imu_frame_, sensor_frame_));
    }
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = current_time_;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = sensor_frame_;
    odom_msg.pose.pose.position.x = T_O_S_.getOrigin().x();
    odom_msg.pose.pose.position.y = T_O_S_.getOrigin().y();
    odom_msg.pose.pose.position.z = T_O_S_.getOrigin().z();
    odom_msg.pose.pose.orientation.w = T_O_S_.getRotation().w();
    odom_msg.pose.pose.orientation.x = T_O_S_.getRotation().x();
    odom_msg.pose.pose.orientation.y = T_O_S_.getRotation().y();
    odom_msg.pose.pose.orientation.z = T_O_S_.getRotation().z();
    odom_pub_.publish(odom_msg);
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Time current_time_;
  tf::Transform T_O_S_;
  std::string odom_frame_;
  std::string sensor_frame_;
  std::string imu_frame_;
  std::string pointcloud_frame_;
  tf::Transform T_I_S_;

  tf::TransformBroadcaster tf_broadcaster_;
  ros::Publisher odom_pub_;

  ros::Timer tf_timer_;

  std::mutex pose_update_mutex_;

  bool tfFromCvMat(cv::Mat position, tf::Matrix3x3* rotation,
                   tf::Vector3* translation) {
    CHECK(rotation != nullptr);
    CHECK(translation != nullptr);
    *rotation =
        tf::Matrix3x3(position.at<float>(0, 0), position.at<float>(0, 1),
                      position.at<float>(0, 2), position.at<float>(1, 0),
                      position.at<float>(1, 1), position.at<float>(1, 2),
                      position.at<float>(2, 0), position.at<float>(2, 1),
                      position.at<float>(2, 2));

    *translation =
        tf::Vector3(position.at<float>(0, 3), position.at<float>(1, 3),
                    position.at<float>(2, 3));

    return true;
  }

  bool tfFromStdVector(std::vector<float> position, tf::Matrix3x3* rotation,
                       tf::Vector3* translation) {
    CHECK(rotation != nullptr);
    CHECK(translation != nullptr);
    *rotation = tf::Matrix3x3(position[0], position[1], position[2],
                              position[4], position[5], position[6],
                              position[8], position[9], position[10]);

    *translation = tf::Vector3(position[3], position[7], position[11]);

    return true;
  }

  bool tfFromEigen(Eigen::Matrix4d position, tf::Matrix3x3* rotation,
                   tf::Vector3* translation) {
    CHECK(rotation != nullptr);
    CHECK(translation != nullptr);
    *rotation = tf::Matrix3x3(position(0, 0), position(0, 1), position(0, 2),
                              position(1, 0), position(1, 1), position(1, 2),
                              position(2, 0), position(2, 1), position(2, 2));

    *translation = tf::Vector3(position(0, 3), position(1, 3), position(2, 3));

    return true;
  }

  tf::Transform TransformFromEigen(Eigen::Matrix4d position_mat) {
    tf::Matrix3x3 tf_camera_rotation;

    tf::Vector3 tf_camera_translation;

    tfFromEigen(position_mat, &tf_camera_rotation, &tf_camera_translation);

    return TransformFromTf(tf_camera_rotation, tf_camera_translation);
  }

  tf::Transform TransformFromCvMat(cv::Mat position_mat) {
    tf::Matrix3x3 tf_camera_rotation;

    tf::Vector3 tf_camera_translation;

    tfFromCvMat(position_mat, &tf_camera_rotation, &tf_camera_translation);

    return TransformFromTf(tf_camera_rotation, tf_camera_translation);
  }

  tf::Transform TransformFromTf(tf::Matrix3x3 rotation,
                                tf::Vector3 translation) {
    // Coordinate transformation matrix from orb coordinate system to ros
    // coordinate system
    const tf::Matrix3x3 tf_orb_to_ros(0, 0, 1, -1, 0, 0, 0, -1, 0);

    // Transform from orb coordinate system to ros coordinate system on camera
    // coordinates
    rotation = tf_orb_to_ros * rotation;
    translation = tf_orb_to_ros * translation;

    // Inverse matrix
    rotation = rotation.transpose();
    translation = -(rotation * translation);

    // Transform from orb coordinate system to ros coordinate system on map
    // coordinates
    rotation = tf_orb_to_ros * rotation;
    translation = tf_orb_to_ros * translation;

    return tf::Transform(rotation, translation);
  }
};

}  // namespace mod
}  // namespace coxgraph

#endif  // COXGRAPH_MOD_TF_PUBLISHER_H_
