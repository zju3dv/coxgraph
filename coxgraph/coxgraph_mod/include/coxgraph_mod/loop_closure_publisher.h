#ifndef COXGRAPH_MOD_LOOP_CLOSURE_PUBLISHER_H_
#define COXGRAPH_MOD_LOOP_CLOSURE_PUBLISHER_H_

#include <coxgraph_msgs/LoopClosure.h>
#include <coxgraph_msgs/MapFusion.h>
#include <coxgraph_msgs/NeedToFuseSrv.h>
#include <eigen_conversions/eigen_msg.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "coxgraph_mod/common.h"

namespace coxgraph {
namespace mod {

class LoopClosurePublisher {
 public:
  typedef std::shared_ptr<LoopClosurePublisher> Ptr;

  LoopClosurePublisher(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private,
                       bool server_mode = false)
      : nh_(nh), nh_private_(nh_private) {
    nh_private_.param("num_agents", client_number_, -1);

    nh_private_.param<std::string>("loop_closure_topic_prefix",
                                   loop_closure_topic_prefix_,
                                   "loop_closure_out_");
    nh_private_.param<std::string>("need_to_fuse_srv_name",
                                   need_to_fuse_srv_name_, "need_to_fuse");
    need_to_fuse_client_ =
        nh_private_.serviceClient<coxgraph_msgs::NeedToFuseSrv>(
            need_to_fuse_srv_name_);

    if (client_number_ == -1) {
      loop_closure_pub_.emplace(
          -1, nh_private_.advertise<coxgraph_msgs::LoopClosure>(
                  "loop_closure_out", 10, true));
    } else {
      for (int i = 0; i < client_number_; i++) {
        loop_closure_pub_.emplace(
            i, nh_private_.advertise<coxgraph_msgs::LoopClosure>(
                   "loop_closure_out_" + std::to_string(i), 10, true));
      }
    }

    for (int i = 0; i < client_number_; i++) {
      for (int j = 0; j < i + 1; j++) {
        need_to_fuse_map_.emplace(std::make_pair(i, j), true);
      }
    }
  }

  ~LoopClosurePublisher() = default;

  bool publishLoopClosure(size_t from_client_id, double from_timestamp,
                          size_t to_client_id, double to_timestamp,
                          geometry_msgs::Quaternion rotation,
                          geometry_msgs::Vector3 transform) {
    coxgraph_msgs::MapFusion map_fusion_msg;
    map_fusion_msg.from_client_id = from_client_id;
    map_fusion_msg.from_timestamp = ros::Time(from_timestamp);
    map_fusion_msg.to_client_id = to_client_id;
    map_fusion_msg.to_timestamp = ros::Time(to_timestamp);
    map_fusion_msg.transform.rotation = rotation;
    map_fusion_msg.transform.translation = transform;

    if (from_client_id != to_client_id) {
      ROS_INFO(
          "Map Fusion Message Published, from client %d time %d, to client "
          "%d time %d ",
          static_cast<int>(from_client_id),
          static_cast<int>(map_fusion_msg.from_timestamp.toSec()),
          static_cast<int>(to_client_id),
          static_cast<int>(map_fusion_msg.to_timestamp.toSec()));
      if (!loop_closure_pub_.count(-1))
        loop_closure_pub_.emplace(
            -1, nh_private_.advertise<coxgraph_msgs::MapFusion>("map_fusion",
                                                                10, true));
      loop_closure_pub_[-1].publish(map_fusion_msg);
    } else {
      ROS_INFO(
          "Loop Closure Message Published, from client %d time %d, to client "
          "%d time %d ",
          static_cast<int>(from_client_id),
          static_cast<int>(map_fusion_msg.from_timestamp.toSec()),
          static_cast<int>(to_client_id),
          static_cast<int>(map_fusion_msg.to_timestamp.toSec()));
      publishLoopClosure(from_client_id, from_timestamp, to_timestamp, rotation,
                         transform);
    }
    return true;
  }
  bool publishLoopClosure(size_t from_client_id, double from_timestamp,
                          size_t to_client_id, double to_timestamp,
                          Eigen::Matrix4d T_A_B) {
    return publishLoopClosure(from_client_id, from_timestamp, to_client_id,
                              to_timestamp, toGeoQuat(T_A_B), toGeoVec3(T_A_B));
  }

  bool publishLoopClosure(size_t from_client_id, double from_timestamp,
                          size_t to_client_id, double to_timestamp, cv::Mat R,
                          cv::Mat t) {
    return publishLoopClosure(from_client_id, from_timestamp, to_client_id,
                              to_timestamp, toGeoQuat(R), toGeoVec3(t));
  }

  bool publishLoopClosure(CliId cid, double from_timestamp, double to_timestamp,
                          geometry_msgs::Quaternion rotation,
                          geometry_msgs::Vector3 transform) {
    coxgraph_msgs::LoopClosure loop_closure_msg;
    loop_closure_msg.from_timestamp = ros::Time(from_timestamp);
    loop_closure_msg.to_timestamp = ros::Time(to_timestamp);
    loop_closure_msg.transform.rotation = rotation;
    loop_closure_msg.transform.translation = transform;

    loop_closure_pub_[cid].publish(loop_closure_msg);
    ROS_INFO_STREAM("Published loop closure msg");

    return true;
  }

  bool publishLoopClosure(CliId cid, const double& from_timestamp,
                          const double& to_timestamp, Eigen::Matrix4d T_A_B) {
    return publishLoopClosure(cid, from_timestamp, to_timestamp,
                              toGeoQuat(T_A_B), toGeoVec3(T_A_B));
  }

  bool publishLoopClosure(CliId cid, const double& from_timestamp,
                          const double& to_timestamp, cv::Mat R, cv::Mat t) {
    return publishLoopClosure(cid, from_timestamp, to_timestamp, toGeoQuat(R),
                              toGeoVec3(t));
  }

  bool needToFuse(CliId cid_a, CliId cid_b, ros::Time time = ros::Time::now()) {
    coxgraph_msgs::NeedToFuseSrv need_to_fuse_srv;
    need_to_fuse_srv.request.cid_a = cid_a;
    need_to_fuse_srv.request.cid_b = cid_b;
    need_to_fuse_srv.request.time = time;
    if (need_to_fuse_client_.call(need_to_fuse_srv))
      return need_to_fuse_srv.response.need_to_fuse;
    else
      return false;
  }

  bool needToFuseCached(CliId cid_a, CliId cid_b) {
    return need_to_fuse_map_[std::make_pair(cid_a, cid_b)];
  }

  void updateNeedToFuse() {
    for (auto& kv : need_to_fuse_map_) {
      kv.second = needToFuse(kv.first.first, kv.first.second);
    }
  }

 private:
  geometry_msgs::Quaternion toGeoQuat(cv::Mat R) {
    tf2::Matrix3x3 tf2_rot(
        R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
        R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
        R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2));
    tf2::Quaternion tf2_quaternion;
    tf2_rot.getRotation(tf2_quaternion);
    return tf2::toMsg(tf2_quaternion);
  }

  geometry_msgs::Quaternion toGeoQuat(Eigen::Matrix4d T_A_B) {
    tf2::Matrix3x3 tf2_rot(T_A_B(0, 0), T_A_B(0, 1), T_A_B(0, 2), T_A_B(1, 0),
                           T_A_B(1, 1), T_A_B(1, 2), T_A_B(2, 0), T_A_B(2, 1),
                           T_A_B(2, 2));
    tf2::Quaternion tf2_quaternion;
    tf2_rot.getRotation(tf2_quaternion);
    return tf2::toMsg(tf2_quaternion);
  }

  geometry_msgs::Vector3 toGeoVec3(Eigen::Matrix4d T_A_B) {
    geometry_msgs::Vector3 transform;
    transform.x = T_A_B(0, 3);
    transform.y = T_A_B(1, 3);
    transform.z = T_A_B(2, 3);
    return transform;
  }

  geometry_msgs::Vector3 toGeoVec3(cv::Mat t) {
    geometry_msgs::Vector3 transform;
    transform.x = t.at<float>(0);
    transform.y = t.at<float>(1);
    transform.z = t.at<float>(2);
    return transform;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  int client_number_;

  std::string loop_closure_topic_prefix_;
  std::string need_to_fuse_srv_name_;
  std::map<CliId, ros::Publisher> loop_closure_pub_;
  ros::ServiceClient need_to_fuse_client_;

  std::map<std::pair<CliId, CliId>, bool> need_to_fuse_map_;
};

}  // namespace mod
}  // namespace coxgraph

#endif  // COXGRAPH_MOD_LOOP_CLOSURE_PUBLISHER_H_
