#ifndef COXGRAPH_MAP_COMM_MESH_CONVERTER_H_
#define COXGRAPH_MAP_COMM_MESH_CONVERTER_H_

#include <coxgraph/common.h>
#include <minkindr_conversions/kindr_msg.h>
#include <nav_msgs/Path.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <voxblox/core/common.h>
#include <voxblox_msgs/Mesh.h>
#include <voxblox_ros/tsdf_server.h>

#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace voxblox {
typedef std::shared_ptr<Pointcloud> PointcloudPtr;
typedef std::shared_ptr<Colors> ColorsPtr;
class MeshConverter {
 public:
  struct Config {
    float voxel_size = 0.20;
    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Mesh Converter using Config:" << std::endl
        << "  voxel_size: " << v.voxel_size << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };

  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private) {
    Config config;
    nh_private.param<float>("interpolate_voxel_size", config.voxel_size,
                            config.voxel_size);
    return config;
  }

  typedef std::shared_ptr<MeshConverter> Ptr;
  typedef kindr::minimal::PositionTemplate<FloatingPoint> Position;

  explicit MeshConverter(const ros::NodeHandle nh_private)
      : config_(getConfigFromRosParam(nh_private)) {
    LOG(INFO) << config_;
    T_odom_submap_.setIdentity();
  }

  ~MeshConverter() = default;

  inline void setMesh(const voxblox_msgs::Mesh& mesh) {
    if (mesh.trajectory.poses.empty()) {
      LOG_IF(ERROR, mesh.trajectory.poses.empty())
          << "received a mesh with empty trajectory";
      return;
    }
    mesh_ = mesh;
    setTrajectory(mesh.trajectory);
  }

  inline void setTrajectory(const nav_msgs::Path& path) {
    for (auto const& pose : path.poses) {
      coxgraph::TransformationD T_Sm_C;
      tf::poseMsgToKindr(pose.pose, &T_Sm_C);
      T_G_C_.emplace_back(pose.header.stamp, T_Sm_C.cast<FloatingPoint>());
    }
    //  const size_t middle_id = T_G_C_.size() / 2;
    //  T_odom_submap_ = TsdfServer::gravityAlignPose(T_G_C_[middle_id].second);
  }

  bool convertToPointCloud(
      pcl::PointCloud<pcl::PointXYZRGB>* recovered_pointcloud) {
    CHECK(recovered_pointcloud != nullptr);
    recovered_pointcloud->clear();
    if (mesh_.mesh_blocks.empty()) return false;
    timing::Timer recovered_poincloud_timer("recover_pointcloud");
    Pointcloud triangle;
    voxblox::Colors colors;

    LOG(INFO) << "receive mesh blocks: " << mesh_.mesh_blocks.size();
    int n = 0, n_colors = 0;
    size_t vertex_index = 0u;
    for (auto const& mesh_block : mesh_.mesh_blocks) {
      if (mesh_block.history.empty()) continue;
      CHECK_EQ(mesh_block.x.size() / 3, mesh_block.history.size());
      const BlockIndex index(mesh_block.index[0], mesh_block.index[1],
                             mesh_block.index[2]);

      // translate vertex data from message to voxblox mesh
      for (size_t i = 0; i < mesh_block.x.size(); ++i) {
        // Each vertex is given as its distance from the blocks origin in units
        // of (2*block_size), see mesh_vis.h for the slightly convoluted
        // justification of the 2.
        constexpr float point_conv_factor =
            2.0f / std::numeric_limits<uint16_t>::max();
        const float mesh_x =
            (static_cast<float>(mesh_block.x[i]) * point_conv_factor +
             static_cast<float>(index[0])) *
            mesh_.block_edge_length;
        const float mesh_y =
            (static_cast<float>(mesh_block.y[i]) * point_conv_factor +
             static_cast<float>(index[1])) *
            mesh_.block_edge_length;
        const float mesh_z =
            (static_cast<float>(mesh_block.z[i]) * point_conv_factor +
             static_cast<float>(index[2])) *
            mesh_.block_edge_length;

        auto history = mesh_block.history[i / 3];
        // CHECK_EQ(history.history.size() % 2, 0);

        triangle.emplace_back(mesh_x, mesh_y, mesh_z);
        colors.emplace_back(mesh_block.r[i], mesh_block.g[i], mesh_block.b[i]);
        pcl::PointXYZRGB recovered_point;
        recovered_point.x = mesh_x;
        recovered_point.y = mesh_y;
        recovered_point.z = mesh_z;
        recovered_point.r = mesh_block.r[i];
        recovered_point.g = mesh_block.g[i];
        recovered_point.b = mesh_block.b[i];
        recovered_pointcloud->push_back(recovered_point);
        if (triangle.size() == 3) {
          if (mesh_block.r[i] * mesh_block.g[i] * mesh_block.b[i] > 0 &&
              mesh_block.r[i - 1] * mesh_block.g[i - 1] * mesh_block.b[i - 1] >
                  0 &&
              mesh_block.r[i - 2] * mesh_block.g[i - 2] * mesh_block.b[i - 2] >
                  0) {
            n_colors++;
          }
          Pointcloud interp_pts;
          Colors interp_colors;
          interpolateTriangle(triangle, colors, &interp_pts, &interp_colors);

          voxblox_msgs::ObsHistory recovered_history;
          for (size_t i = 0; i < history.history.size(); i += 2) {
            for (size_t j = history.history[i]; j <= history.history[i + 1];
                 j++)
              recovered_history.history.emplace_back(j);
          }

          for (size_t hi = 0; hi < recovered_history.history.size(); hi++) {
            auto stamp = recovered_history.history[hi];
            if (!pointcloud_.count(stamp))
              pointcloud_.emplace(
                  stamp, std::make_pair(new Pointcloud(), new Colors()));
            pointcloud_[stamp].first->insert(pointcloud_[stamp].first->end(),
                                             triangle.begin(), triangle.end());
            pointcloud_[stamp].first->insert(pointcloud_[stamp].first->end(),
                                             interp_pts.begin(),
                                             interp_pts.end());
            pointcloud_[stamp].second->insert(pointcloud_[stamp].second->end(),
                                              colors.begin(), colors.end());
            pointcloud_[stamp].second->insert(pointcloud_[stamp].second->end(),
                                              interp_colors.begin(),
                                              interp_colors.end());
          }
          triangle.clear();
          colors.clear();
        }
      }
      n++;
    }

    LOG(INFO) << "processed " << n;

    recovered_poincloud_timer.Stop();

    return true;
  }

  void clear() {
    // Clear point clouds
    for (auto const& pointcloud : pointcloud_) {
      pointcloud.second.first->clear();
      pointcloud.second.second->clear();
    }
    pointcloud_.clear();
    T_G_C_.clear();
    mesh_.mesh_blocks.clear();
    T_odom_submap_.setIdentity();
  }

  bool getNextPointcloud(int* i, Transformation* T_G_C,
                         PointcloudPtr* pointcloud, ColorsPtr* colors) {
    CHECK(*pointcloud != nullptr);
    CHECK(*colors != nullptr);
    if (*i >= T_G_C_.size()) return false;
    timing::Timer next_pointcloud_timer("next_pointcloud");
    CHECK(pointcloud);
    *T_G_C = T_G_C_[*i].second;
    auto id =
        T_G_C_[*i].first == T_G_C_.begin()->first
            ? 0
            : std::round((T_G_C_[*i].first - T_G_C_.begin()->first).toSec() /
                         0.05);
    if (!pointcloud_.count(id))
      pointcloud_.emplace(id, std::make_pair(new Pointcloud(), new Colors()));
    // Point cloud is actually T_Submap_P; to get T_C_P, need to get T_Submap_C;
    auto T_Submap_C = T_odom_submap_.inverse() * (*T_G_C);
    transformPointcloud(T_Submap_C.inverse(), *(pointcloud_[id].first),
                        pointcloud->get());
    *colors = pointcloud_[id].second;
    (*i)++;
    next_pointcloud_timer.Stop();
    return true;
  }

  void interpolateTriangle(const Pointcloud& triangle, const Colors& colors,
                           Pointcloud* interp_pc, Colors* interp_colors) {
    CHECK_EQ(triangle.size(), colors.size());
    timing::Timer interpolate_timer("interpolate_triangle");
    // TODO(mikexyl): a really stupid interpolation, but should do the work
    Pointcloud interp_pts_e01, interp_pts_e02, interp_pts_e12;
    Colors interp_colors_e01, interp_colors_e02, interp_colors_e12;
    Point p0 = triangle[0], p1 = triangle[1], p2 = triangle[2];

    Point t_p0_p1 = p1 - p0;
    Point t_p0_p2 = p2 - p0;
    Point t_p1_p2 = p2 - p1;

    for (float dist = config_.voxel_size; dist < t_p0_p1.norm();
         dist += config_.voxel_size) {
      interp_pts_e01.emplace_back(p0 + t_p0_p1 / t_p0_p1.norm() * dist);
      interp_colors_e01.emplace_back(
          Color::blendTwoColors(colors[0], (1 - dist / t_p0_p1.norm()),
                                colors[1], dist / t_p0_p1.norm()));
    }
    for (float dist = config_.voxel_size; dist < t_p0_p2.norm();
         dist += config_.voxel_size) {
      interp_pts_e02.emplace_back(p0 + t_p0_p2 / t_p0_p2.norm() * dist);
      interp_colors_e02.emplace_back(
          Color::blendTwoColors(colors[0], (1 - dist / t_p0_p2.norm()),
                                colors[1], dist / t_p0_p2.norm()));
    }
    for (float dist = config_.voxel_size; dist < t_p1_p2.norm();
         dist += config_.voxel_size) {
      interp_pts_e12.emplace_back(p1 + t_p1_p2 / t_p1_p2.norm() * dist);
      interp_colors_e12.emplace_back(
          Color::blendTwoColors(colors[1], (1 - dist / t_p1_p2.norm()),
                                colors[2], dist / t_p1_p2.norm()));
    }

    interp_pts_e01.emplace_back((p0 + p1 + p2) / 3);
    interp_colors_e01.emplace_back(Color::blendTwoColors(
        colors[2], 1 / 3.0,
        Color::blendTwoColors(colors[0], 0.5, colors[1], 0.5), 2 / 3.0));

    interp_pts_e01.insert(interp_pts_e01.end(), interp_pts_e02.begin(),
                          interp_pts_e02.end());
    interp_pts_e01.insert(interp_pts_e01.end(), interp_pts_e12.begin(),
                          interp_pts_e12.end());

    *interp_pc = interp_pts_e01;
    interp_colors->insert(interp_colors->end(), interp_colors_e01.begin(),
                          interp_colors_e01.end());
    interp_colors->insert(interp_colors->end(), interp_colors_e02.begin(),
                          interp_colors_e02.end());
    interp_colors->insert(interp_colors->end(), interp_colors_e12.begin(),
                          interp_colors_e12.end());

    interpolate_timer.Stop();
  }

 private:
  Config config_;

  voxblox_msgs::Mesh mesh_;
  AlignedVector<std::pair<ros::Time, Transformation>> T_G_C_;
  Transformation T_odom_submap_;

  std::map<uint8_t, std::pair<PointcloudPtr, ColorsPtr>> pointcloud_;
};
}  // namespace voxblox

#endif  //  COXGRAPH_MAP_COMM_MESH_CONVERTER_H_
