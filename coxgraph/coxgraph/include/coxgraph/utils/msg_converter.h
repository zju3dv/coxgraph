#ifndef COXGRAPH_UTILS_MSG_CONVERTER_H_
#define COXGRAPH_UTILS_MSG_CONVERTER_H_

// #include <open3d/geometry/TriangleMesh.h>
#include <Open3D/Geometry/TriangleMesh.h>
#include <cblox_msgs/MapLayer.h>
#include <cblox_ros/submap_conversions.h>
#include <coxgraph_msgs/BoundingBox.h>
#include <coxgraph_msgs/ClientSubmap.h>
#include <coxgraph_msgs/ClientSubmapSrvResponse.h>
#include <coxgraph_msgs/MapFusion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <voxblox_msgs/Layer.h>
#include <voxblox_msgs/LayerWithTrajectory.h>
#include <voxblox_ros/conversions.h>
#include <voxgraph_msgs/LoopClosure.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "coxgraph/common.h"

namespace coxgraph {
namespace utils {

inline cblox_msgs::MapLayer tsdfEsdfMsgfromClientSubmap(
    const CliSm& submap, const std::string& frame_id) {
  cblox_msgs::MapLayer submap_esdf_msg;
  cblox::serializeSubmapToMsg<cblox::TsdfEsdfSubmap>(submap, &submap_esdf_msg);
  submap_esdf_msg.map_header.pose_estimate.frame_id = frame_id;
  return submap_esdf_msg;
}

inline cblox_msgs::MapLayer tsdfMsgfromClientSubmap(
    const CliSm& submap, const std::string& frame_id) {
  cblox_msgs::MapLayer submap_tsdf_msg;
  cblox::serializeSubmapToMsg<cblox::TsdfSubmap>(submap, &submap_tsdf_msg);
  submap_tsdf_msg.map_header.pose_estimate.frame_id = frame_id;
  return submap_tsdf_msg;
}

inline coxgraph_msgs::ClientSubmap msgFromCliSubmap(
    const CliSm& submap, const std::string& frame_id) {
  voxblox_msgs::Layer layer_msg;
  voxblox::serializeLayerAsMsg<voxblox::TsdfVoxel>(
      submap.getTsdfMap().getTsdfLayer(), false, &layer_msg);
  voxblox_msgs::LayerWithTrajectory layer_with_trajectory_msg;
  layer_with_trajectory_msg.layer = layer_msg;

  LOG(INFO) << "debug: submap pose history size: "
            << submap.getPoseHistory().size();
  for (auto const& time_pose_kv : submap.getPoseHistory()) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = time_pose_kv.first;
    tf::poseKindrToMsg(time_pose_kv.second.cast<double>(), &pose_msg.pose);
    layer_with_trajectory_msg.trajectory.poses.emplace_back(pose_msg);
  }

  coxgraph_msgs::ClientSubmap cli_submap_msg;
  cli_submap_msg.layer_with_traj = layer_with_trajectory_msg;
  cli_submap_msg.map_header.id = submap.getID();
  cli_submap_msg.map_header.start = submap.getStartTime();
  cli_submap_msg.map_header.end = submap.getEndTime();
  cli_submap_msg.map_header.header.stamp = ros::Time::now();
  tf::poseKindrToMsg(submap.getPose().cast<double>(),
                     &cli_submap_msg.map_header.pose.map_pose);
  cli_submap_msg.map_header.pose.frame_id = frame_id;
  cli_submap_msg.mesh_pointclouds = *submap.mesh_pointcloud_;
  return cli_submap_msg;
}

/**
 * @brief Generate Client Submap from Message
 *
 * @param ser_sm_id
 * @param submap_config
 * @param submap_response
 * @param frame_id
 * @return CliSm::Ptr
 */
inline CliSm::Ptr cliSubmapFromMsg(
    const SerSmId& ser_sm_id, const CliSmConfig& submap_config,
    const coxgraph_msgs::ClientSubmap& submap_msg, std::string* frame_id) {
  CliSm::Ptr submap_ptr(new CliSm(Transformation(), ser_sm_id, submap_config));

  // Naming copied from voxgraph
  for (const geometry_msgs::PoseStamped& pose_stamped :
       submap_msg.layer_with_traj.trajectory.poses) {
    TransformationD T_submap_base_link;
    tf::poseMsgToKindr(pose_stamped.pose, &T_submap_base_link);
    submap_ptr->addPoseToHistory(
        pose_stamped.header.stamp,
        T_submap_base_link.cast<voxblox::FloatingPoint>());
  }

  if (submap_ptr->getPoseHistory().size()) {
    TransformationD submap_pose;
    tf::poseMsgToKindr(submap_msg.map_header.pose.map_pose, &submap_pose);
    submap_ptr->setPose(submap_pose.cast<voxblox::FloatingPoint>());
    *frame_id = submap_msg.map_header.pose.frame_id;

    // Deserialize the submap TSDF
    if (!voxblox::deserializeMsgToLayer(
            submap_msg.layer_with_traj.layer,
            submap_ptr->getTsdfMapPtr()->getTsdfLayerPtr())) {
      LOG(FATAL)
          << "Received a submap msg with an invalid TSDF. Skipping submap.";
    }
    submap_ptr->finishSubmap();
    *submap_ptr->mesh_pointcloud_ = submap_msg.mesh_pointclouds;
  }

  return submap_ptr;
}

inline CliSm::Ptr cliSubmapFromMsg(
    const SerSmId& ser_sm_id, const CliSmConfig& submap_config,
    const coxgraph_msgs::ClientSubmapSrvResponse& submap_response,
    std::string* frame_id) {
  return cliSubmapFromMsg(ser_sm_id, submap_config, submap_response.submap,
                          frame_id);
}

inline voxgraph_msgs::LoopClosure fromMapFusionMsg(
    const coxgraph_msgs::MapFusion& map_fusion_msg) {
  CHECK_EQ(map_fusion_msg.from_client_id, map_fusion_msg.to_client_id);
  voxgraph_msgs::LoopClosure loop_closure_msg;
  loop_closure_msg.from_timestamp = map_fusion_msg.from_timestamp;
  loop_closure_msg.to_timestamp = map_fusion_msg.to_timestamp;
  loop_closure_msg.transform = map_fusion_msg.transform;
  return loop_closure_msg;
}

inline coxgraph_msgs::BoundingBox msgFromBb(const BoundingBox& bounding_box) {
  coxgraph_msgs::BoundingBox bb_msg;
  bb_msg.min[0] = bounding_box.min[0];
  bb_msg.min[1] = bounding_box.min[1];
  bb_msg.min[2] = bounding_box.min[2];
  bb_msg.max[0] = bounding_box.max[0];
  bb_msg.max[1] = bounding_box.max[1];
  bb_msg.max[2] = bounding_box.max[2];
  return bb_msg;
}

inline uint64_t sizeOfMsg(const coxgraph_msgs::ClientSubmap& msg) {
  uint64_t total_size = 0;
  for (auto const& block : msg.layer_with_traj.layer.blocks) {
    total_size += 3 * sizeof(block.x_index);
    if (block.data.size())
      total_size += block.data.size() * sizeof(block.data[0]);
  }
  total_size += sizeof(msg.layer_with_traj.layer.voxel_size);
  total_size += sizeof(msg.layer_with_traj.layer.voxels_per_side);
  total_size += sizeof(msg.layer_with_traj.layer.layer_type);
  total_size += sizeof(msg.layer_with_traj.layer.action);
  total_size += sizeof(msg.layer_with_traj.trajectory.header);
  if (msg.layer_with_traj.trajectory.poses.size())
    total_size += msg.layer_with_traj.trajectory.poses.size() *
                  sizeof(msg.layer_with_traj.trajectory.poses[0]);
  total_size += sizeof(msg.map_header);

  return total_size;
}

inline CIdCSIdPair resolveSubmapFrame(std::string frame_id) {
  frame_id.erase(0, 7);
  size_t pos = frame_id.find_last_of('_');
  CliSmId csid = std::stoi(frame_id.substr(0, pos));
  CliId cid = std::stoi(frame_id.substr(pos + 1, frame_id.size()));
  return std::make_pair(cid, csid);
}

inline Eigen::Vector3d getColor(Eigen::Vector3d ori_color, int color_mode,
                                CliId cid) {
  Eigen::Vector3d color;
  if (color_mode == 0)
    return ori_color;
  else if (color_mode == 2) {
    color = ori_color;
    if (cid == 0) {
      color[0] *= 1.0;
      color[1] *= 0.5;
      color[2] *= 0.5;
    } else if (cid == 1) {
      color[0] *= 0.5;
      color[1] *= 1.0;
      color[2] *= 0.5;
    } else if (cid == 2) {
      color[0] *= 0.5;
      color[1] *= 0.5;
      color[2] *= 1.0;
    }
    return color;
  }
  return ori_color;
}

inline std::shared_ptr<open3d::geometry::TriangleMesh> o3dMeshFromMsg(
    const sensor_msgs::PointCloud2& pointcloud2_msg, int color_mode = 0,
    CliId cid = 0) {
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3d> colors;
  std::vector<Eigen::Vector3i> indices;

  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  pcl::fromROSMsg(pointcloud2_msg, pointcloud);
  if (pointcloud.empty()) return nullptr;

  CHECK_EQ(pointcloud.points.size() % 3, 0);
  for (size_t i = 0; i < pointcloud.size() - 3; i += 3) {
    auto point = pointcloud[i];
    vertices.emplace_back(point.x, point.y, point.z);
    colors.emplace_back(getColor(
        Eigen::Vector3d(point.r / 255.0, point.g / 255.0, point.b / 255.0),
        color_mode, cid));
    point = pointcloud[i + 1];
    vertices.emplace_back(point.x, point.y, point.z);
    colors.emplace_back(getColor(
        Eigen::Vector3d(point.r / 255.0, point.g / 255.0, point.b / 255.0),
        color_mode, cid));
    point = pointcloud[i + 2];
    vertices.emplace_back(point.x, point.y, point.z);
    colors.emplace_back(getColor(
        Eigen::Vector3d(point.r / 255.0, point.g / 255.0, point.b / 255.0),
        color_mode, cid));

    indices.emplace_back(i, i + 1, i + 2);
  }
  CHECK_EQ(vertices.size() / indices.size(), 3);

  std::shared_ptr<open3d::geometry::TriangleMesh> o3d_mesh(
      new open3d::geometry::TriangleMesh(vertices, indices));

  Eigen::Vector3d color;
  if (color_mode == 1) {
    switch (cid) {
      case 0:
        color[0] = 1.0;
        color[1] = 0;
        color[2] = 0;
        break;
      case 1:
        color[0] = 0;
        color[1] = 1.0;
        color[2] = 0;
        break;
      case 2:
        color[0] = 0;
        color[1] = 0;
        color[2] = 1.0;
        break;
    }
    for (size_t i = 0; i < o3d_mesh->vertices_.size(); i++)
      o3d_mesh->vertex_colors_.emplace_back(color);
  } else {
    o3d_mesh->vertex_colors_ = colors;
  }

  return o3d_mesh;
}

}  // namespace utils
}  // namespace coxgraph

#endif  // COXGRAPH_UTILS_MSG_CONVERTER_H_
