#ifndef COXGRAPH_MAP_COMM_TSDF_RECOVER_H_
#define COXGRAPH_MAP_COMM_TSDF_RECOVER_H_

#include <cblox_msgs/MapPoseUpdates.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox_msgs/LayerWithTrajectory.h>
#include <voxblox_ros/ros_params.h>
#include <voxblox_ros/transformer.h>
#include <voxblox_ros/tsdf_server.h>

#include <map>
#include <string>

#include "coxgraph/map_comm/mesh_converter.h"
namespace voxblox {
class TsdfRecover : public TsdfServer {
 public:
  struct Config {
    Config() : use_tf_submap_pose(false) {}
    bool use_tf_submap_pose;

    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Tsdf Recover using Config:" << std::endl
        << "  Use TF Submap Pose: "
        << static_cast<std::string>(v.use_tf_submap_pose ? "enabled"
                                                         : "disabled")
        << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };

  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private) {
    Config config;
    nh_private.param<bool>("use_tf_submap_pose", config.use_tf_submap_pose,
                           config.use_tf_submap_pose);
    return config;
  }

  TsdfRecover(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : nh_(nh),
        nh_private_(nh_private),
        config_(getConfigFromRosParam(nh_private)),
        TsdfServer(nh, nh_private, getTsdfMapConfigFromRosParam(nh_private),
                   getTsdfIntegratorConfigFromRosParam(nh_private),
                   getMeshIntegratorConfigFromRosParam(nh_private)) {
    mesh_converter_.reset(new MeshConverter(nh_private_));

    subscribeToTopics();
    advertiseTopics();
  }

  ~TsdfRecover() = default;

  bool processMesh(const voxblox_msgs::Mesh& mesh_msg,
                   voxblox_msgs::LayerWithTrajectory* layer_msg,
                   pcl::PointCloud<pcl::PointXYZRGB>* recovered_pointcloud) {
    tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
    timing::Timer mesh_process_timer("mesh_process");
    mesh_converter_->setMesh(mesh_msg);
    mesh_converter_->convertToPointCloud(recovered_pointcloud);

    Transformation T_G_C;
    PointcloudPtr points_C(new Pointcloud());
    int i = 0;
    ColorsPtr colors(new Colors());
    while (mesh_converter_->getNextPointcloud(&i, &T_G_C, &points_C, &colors)) {
      if (points_C->empty()) continue;

      timing::Timer integrator("integrate");
      tsdf_integrator_->integratePointCloud(T_G_C, *points_C, *colors, false);
      integrator.Stop();

      //    if (frame_pointcloud_pub_.getNumSubscribers() > 0) {
      //      pcl::PointCloud<pcl::PointXYZ> pointcloud_msg;
      //      pointcloud_msg.header.frame_id = world_frame_;
      //      Pointcloud points_G;
      //      transformPointcloud(T_G_C, points_C, &points_G);
      //      pointcloudToPclXYZ(points_G, &pointcloud_msg);
      //      frame_pointcloud_pub_.publish(pointcloud_msg);
      //    }
    }

    mesh_converter_->clear();

    mesh_process_timer.Stop();

    LOG(INFO) << "layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize();
    ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());

    serializeLayerAsMsg(tsdf_map_->getTsdfLayer(), false, &layer_msg->layer);
    layer_msg->trajectory = mesh_msg.trajectory;

    return true;
  }

 private:
  void subscribeToTopics() {
    mesh_sub_ = nh_private_.subscribe("mesh_with_history", 10,
                                      &TsdfRecover::meshCallback, this);
    if (!config_.use_tf_submap_pose)
      submap_pose_sub_ = nh_private_.subscribe(
          "submap_poses", 10, &TsdfRecover::submapPoseCallback, this);
    else
      LOG(FATAL) << "Don't turn on use_tf_submap_pose, bug unfix";
  }
  void advertiseTopics() {
    recovered_pointcloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>(
        "mesh_pointcloud", 1, true);
    frame_pointcloud_pub_ =
        nh_private_.advertise<pcl::PointCloud<pcl::PointXYZ>>(
            "in_fov_pointcloud", 1, true);
  }

  void meshCallback(const voxblox_msgs::Mesh& mesh_msg) {
    pcl::PointCloud<pcl::PointXYZRGB> recovered_pointcloud;
    voxblox_msgs::LayerWithTrajectory layer_msg;
    processMesh(mesh_msg, &layer_msg, &recovered_pointcloud);
    sensor_msgs::PointCloud2 recovered_pointcloud_msg;
    pcl::toROSMsg<pcl::PointXYZRGB>(recovered_pointcloud,
                                 recovered_pointcloud_msg);
    recovered_pointcloud_msg.header.stamp = mesh_msg.header.stamp;
    layer_msg.header.stamp = mesh_msg.header.stamp;
    tsdf_map_pub_.publish(layer_msg);
    recovered_pointcloud_pub_.publish(recovered_pointcloud_msg);

    if (verbose_) {
      ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
      ROS_INFO_STREAM(
          "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  Config config_;

  ros::Subscriber mesh_sub_;
  ros::Publisher recovered_pointcloud_pub_;
  ros::Publisher frame_pointcloud_pub_;
  ros::Publisher submap_pub_;

  MeshConverter::Ptr mesh_converter_;

  tf::TransformListener tf_listener_;

  ros::Subscriber submap_pose_sub_;
  std::map<int16_t, Transformation> submap_pose_map_;
  void submapPoseCallback(const cblox_msgs::MapPoseUpdates& pose_update_msg) {
    for (auto const& map_header : pose_update_msg.map_headers) {
      kindr::minimal::QuatTransformationTemplate<double> T_G_Sm;
      tf::poseMsgToKindr(map_header.pose_estimate.map_pose, &T_G_Sm);
      auto submap_pose_kv = submap_pose_map_.find(map_header.id);
      if (submap_pose_kv == submap_pose_map_.end()) {
        submap_pose_map_.emplace(map_header.id, T_G_Sm.cast<FloatingPoint>());
      } else {
        submap_pose_kv->second = T_G_Sm.cast<FloatingPoint>();
      }
    }
  }

  bool getSubmapPoseBlocking(int16_t submap_id, Transformation* T_G_Sm) {
    float loop_ms = 10;
    ros::Rate loop(ros::Duration(loop_ms / 1000));
    float wait_time_ms = 0;
    while (!submap_pose_map_.count(submap_id)) {
      loop.sleep();
      wait_time_ms += loop_ms;
      if (wait_time_ms > 500) {
        return false;
      }
    }
    *T_G_Sm = submap_pose_map_.find(submap_id)->second;
    return true;
  }

  constexpr static float kSubmapPoseToleranceMs = 500;
};
}  // namespace voxblox

#endif  //  COXGRAPH_MAP_COMM_TSDF_RECOVER_H_
